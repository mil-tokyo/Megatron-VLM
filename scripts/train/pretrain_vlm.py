# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""
import logging
import math
import os
import sys
from functools import partial

import torch
import torch._dynamo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.pardir)))

from dataloader_provider import train_valid_test_dataloaders_provider
from llava_model_provider import get_image_token_count, model_provider

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel import broadcast_data
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from pretrain_gpt import loss_func

torch._dynamo.config.suppress_errors = True


def model_provider_wrapper(
        pre_process=True, post_process=True,
        add_encoder: bool = True
):
    """
    Wrapper for model provider; this is needed to adjust the arguments of the model provider function.
    Args:
        pre_process: (bool) not used in vlm's model_provider
        post_process: (bool) not used in vlm's model_provider
        add_encoder: (bool) whether to include the vision encoder in the model;
                                    this is required for vlm's model_provider, but not for other models
    Returns:
        (callable) model provider function
    """
    parallel_state.get_pipeline_model_parallel_rank()
    return model_provider(add_encoder=add_encoder, pre_process=pre_process, post_process=post_process)


def get_ltor_masks_and_position_ids(
        data, eod_token,
        reset_position_ids, reset_attention_mask,
        eod_mask_loss, num_image_tokens,
        question_length=None, weights=None
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

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

    if question_length is not None:
        for b in range(micro_batch_size):
            should_mask_len = question_length[b].item() + num_image_tokens
            loss_mask[b, :max(num_image_tokens, should_mask_len)] = 0.0

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
    if weights is not None:
        loss_mask = loss_mask * weights

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""

    args = get_args()

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_text = broadcast_data(["text"], data, torch.int64)["text"]
    data_img = broadcast_data(["img"], data, torch.float32)
    prompt_len = broadcast_data(["prompt_len"], data, torch.int64)["prompt_len"]

    torch.cuda.nvtx.range_pop()

    tokens_ = data_text.long()

    img_raw = data_img['img'].reshape(-1, 3, args.img_h, args.img_w)

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = args.decoder_seq_length - args.seq_length

    tokens = tokens_[:, :text_length].contiguous()
    labels = tokens_[:, :text_length+1].contiguous()

    # make tokens divisible by tensor model parallel size
    # note: input is cut off by 1 token before inputting to the model, so we need to add 1 token here
    if args.sequence_parallel:
        # if args.sequence_parallel:
        # divisible by least common multiple of tensor model parallel size and 16
        divisible_by = math.lcm(parallel_state.get_tensor_model_parallel_world_size(), 16)
        # else:
        #     divisible_by = 16
        if tokens.size(1) % divisible_by != 0:
            remainder = tokens.size(1) % divisible_by
            pad_size = divisible_by - remainder + 1
            if args.add_class_token:
                pad_size -= 1
            # add padding (pad_token_id) to tokens
            tokens = torch.cat([
                tokens,
                torch.full((tokens.size(0), pad_size), args.pad_token_id, device=tokens.device, dtype=tokens.dtype)
            ], dim=1)
            # add padding (pad_token_id) to labels
            labels = torch.cat([
                labels,
                torch.full((labels.size(0), pad_size - 1), args.pad_token_id, device=labels.device, dtype=labels.dtype)
            ], dim=1)

    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens, eod_token,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss,
                                        get_image_token_count(),
                                        question_length=prompt_len)
    torch.cuda.nvtx.range_pop()

    # loss_mask, labels, attention_mask = _preprocess_data_for_llava(loss_mask, labels, attention_mask)

    # tokens = tokens[:, 1:]  # drop image index token
    # loss_mask = loss_mask[:, 1:]
    # make sure loss_mask's size is divisible by tensor model parallel size
    if args.sequence_parallel:
        if loss_mask.size(1) % divisible_by != 0:
            remainder = loss_mask.size(1) % divisible_by
            pad_size = divisible_by - remainder + 1
            if args.add_class_token:
                pad_size -= 1
            # add padding (1) to loss_mask
            loss_mask = torch.cat([
                loss_mask,
                torch.ones(loss_mask.size(0), pad_size, device=loss_mask.device, dtype=loss_mask.dtype)
            ], dim=1)
    loss_mask = loss_mask[:, 1:]  # drop image index token
    return tokens, labels, loss_mask, attention_mask, position_ids, img_raw


def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided,
                    otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(images, tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--valid-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    # group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    # group.add_argument('--language-model-type', type=str, required=True)
    # group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    # group.add_argument("--allow-missing-vision-projection-checkpoint", action="store_true", default=False)
    # group.add_argument("--use-te", action="store_true", default=False)
    # group.add_argument("--img-embedding-idx", type=int, default=0,
    #                    help='Llava specific parameter. Defines at which index'
    #                    'in the language_embedding tensor the image_embeddings'
    #                    'should be inserted')
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]

if __name__ == "__main__":
    torch._logging.set_logs(dynamo=logging.ERROR)

    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        # train_valid_test_datasets_provider,
        train_valid_test_dataloaders_provider,
        model_provider_wrapper,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'HuggingFaceTokenizer'},
        strict_checkpoint_loading=False,
        extra_args_provider=add_multimodal_extra_args,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
