# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
# from megatron.legacy.model import GPTModel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec_llama
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.inference.text_generation_server import MegatronServer
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process
import torch


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    # model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_llama(
        args.num_experts, args.moe_grouped_gemm
    )

    # LLaMA2 specific config overrides
    config.add_bias_linear = False
    config.gated_linear_unit = True
    config.activation_func = F.silu
    config.normalization = 'RMSNorm'
    config.masked_softmax_fusion = False
    config.bias_dropout_fusion = False
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    args.untie_embeddings_and_output_weights = True
    args.position_embedding_type = 'rope'

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,  # 32000
        max_sequence_length=args.max_position_embeddings,  # 4096
        pre_process=pre_process,  # True
        post_process=post_process,  # True
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,  # False
        # parallel_output=True,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,  # False
        position_embedding_type=args.position_embedding_type,  # learned_absolute
        rotary_percent=args.rotary_percent,  # 1.0
    )

    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    # add LLaMA2 specific args
    args.add_position_embedding = False

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        print("Loading checkpoint from: {}".format(args.load))
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0",port=args.port)

    while True:
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        print(f"choice: {choice}")
        # if choice[0].item() == 0:
        if choice.item() == 0:
            try:
                generate_and_post_process(model)
            except ValueError as ve:
                pass
        # elif choice[0].item() == 1:
        elif choice.item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
