# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import argparse
import os
import sys
from datetime import datetime

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

    try:
        from amp_C import multi_tensor_l2norm
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False  # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
                      torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()
    cp_size = args.context_parallel_size
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1):],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
                                     device="cpu", pin_memory=True).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                batch[key] = val

    return batch


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def append_to_progress_log(string, barrier=True):
    """ Append given string to progress log. """
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            'position_ids': data["position_ids"].cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch


def num_floating_points(
        checkpoint_activations_factor: int,
        selective_recompute_factor: int,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        activation_function_factor: int,
        vocab_size: int = 0,
        gqa_group_size: int = 1,
        do_backward: bool = True,
):
    attention_matrix_flops = (4 * batch_size * (seq_len ** 2) * hidden_size) / gqa_group_size * selective_recompute_factor
    logit_layer_flops = 2 * batch_size * seq_len * hidden_size * vocab_size
    qkv_transformations_flops = 2 * 3 * batch_size * seq_len * (hidden_size ** 2)
    post_attention_linear_flops = 2 * batch_size * seq_len * (hidden_size ** 2)
    ffn_flops = activation_function_factor * (intermediate_size / hidden_size) * batch_size * seq_len * (hidden_size ** 2)

    transformer_block_flops = num_layers * checkpoint_activations_factor * (qkv_transformations_flops + attention_matrix_flops + post_attention_linear_flops + ffn_flops)
    total_flops = transformer_block_flops + logit_layer_flops

    if not do_backward:
        # backward pass requires 4x more flops
        # i.e., 2x for gradients and 1x for re-computing activations
        total_flops = total_flops / 4

    return total_flops

def throughput_calculator(
    args: argparse.Namespace,
    iteration_time: float,
    total_iterations: int,
) -> tuple[float, float, int, int]:
    gpus_per_model: int = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())
    batch_size: int = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    samples_per_model: int = batch_size * args.seq_length
    model_replica_count: int = torch.distributed.get_world_size() // gpus_per_model
    elapsed_time_per_iter = iteration_time / total_iterations
    samples_per_second: float = batch_size / elapsed_time_per_iter

    # 1. Calculate TFLOPs for decoder
    # flops calculator
    hidden_size: int = args.hidden_size
    num_layers: int = args.num_layers
    vocab_size: int = args.padded_vocab_size
    intermediate_size: int = args.ffn_hidden_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # SwiGLU: https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/283
    # otherwise it will be 3.
    checkpoint_activations_factor: int = 3
    selective_recompute_factor: int = 1
    if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_activations') and args.recompute_activations:
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'full'):
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'selective'):
        # add forward attention matrix computation & attention over Values (later)
        checkpoint_activations_factor = 3
        selective_recompute_factor = 2

    seq_len: int = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len: int = args.actual_seq_length

    activation_function_factor: int = 4  # GELU
    if args.swiglu:
        activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)
    gqa_group_size: int = 1
    if args.group_query_attention:  # GQA
        gqa_group_size = args.num_attention_heads // args.num_query_groups

    # 2: post-attention linear projection
    # 2 * 3: Key, Query, and Value transformation
    # / gqa_group_size : GQA: Grouped Query Attention (default: gqa_group_size=1)
    # flops_per_iteration: float = checkpoint_activations_factor * ((
    #     (2 + (2 * 3) + activation_function_factor * (intermediate_size / hidden_size)) * batch_size * seq_len * num_layers * (hidden_size**2)
    # ) + (
    #     ((  # Attention matrix & attention over values
    #         4 * batch_size * (seq_len ** 2) * hidden_size
    #     ) / gqa_group_size * selective_recompute_factor
    #     ) +  # noqa: W504
    #     # lm-head: logit layer
    #     2 * batch_size * seq_len * hidden_size * vocab_size)
    # )
    decoder_flops = num_floating_points(
        checkpoint_activations_factor=checkpoint_activations_factor,
        selective_recompute_factor=selective_recompute_factor,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        activation_function_factor=activation_function_factor,
        vocab_size=vocab_size,
        gqa_group_size=gqa_group_size,
        do_backward=False if args.freeze_LM else True,
    )

    # 2. Calculate TFLOPs for encoder
    vision_hidden_size: int = args.vision_hidden_size
    vision_num_layers: int = args.vision_num_layers
    vision_intermediate_size: int = args.vision_ffn_hidden_size
    vision_seq_len = args.patch_dim ** 2 + 1  # 576
    vision_encoder_flops = num_floating_points(
        checkpoint_activations_factor=checkpoint_activations_factor,
        selective_recompute_factor=selective_recompute_factor,
        batch_size=batch_size,
        seq_len=vision_seq_len,
        hidden_size=vision_hidden_size,
        num_layers=vision_num_layers,
        intermediate_size=vision_intermediate_size,
        activation_function_factor=activation_function_factor,
        vocab_size=0,
        gqa_group_size=1,
        do_backward=False,
    )

    # 3. calculate FLOPS for projector
    projector_flops = 2 * (2 + 4) * batch_size * vision_hidden_size * args.projector_hidden_size * vision_seq_len

    # 4. calculate total FLOPs
    total_flops = decoder_flops + vision_encoder_flops + projector_flops
    tflops = total_flops / (elapsed_time_per_iter * args.world_size * (10**12))

    # tflops: float = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))

    return samples_per_second, tflops, samples_per_model, model_replica_count
