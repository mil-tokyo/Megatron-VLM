# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import types

from transformers import AutoConfig

from utils import get_mcore_transformer_block_key, print_memory_usage


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--tensor-model-parallel-size', type=int, default=1, help='Number of tensor model parallel groups')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='Number of pipeline model parallel groups')


def load_llava_args_from_checkpoint(args):
    print("loader_mcore_llava | Loading LLAVA args from checkpoint ... (loader_mcore_llava.py")

    # Read Llama args.
    llava_args_path = os.path.join(args.load, "config.json")
    with open(llava_args_path) as f:
        llava_args = json.load(f)

    # load text config
    text_model_name_or_path = llava_args["text_config"]["_name_or_path"]
    text_config = AutoConfig.from_pretrained(text_model_name_or_path)

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = 4096

    # setup language model arguments
    args.hidden_size = text_config.hidden_size
    args.num_attention_heads = text_config.num_attention_heads
    args.num_layers = text_config.num_hidden_layers
    args.norm_epsilon = llava_args["text_config"]["rms_norm_eps"]
    args.swiglu = True
    args.normalization = "RMSNorm"
    args.ffn_hidden_size = text_config.intermediate_size
    if "num_key_value_heads" in text_config.__dict__:
        # num_key_value_heads is actually in the text_config
        args.group_query_attention = True
        # args.num_query_groups = llava_args["num_key_value_heads"]
        args.num_query_groups = text_config.num_key_value_heads

    # disable dropout
    args.attention_dropout = 0.0
    args.hidden_dropout = 0.0

    # setup vision model arguments
    args.vision_hidden_size = llava_args["vision_config"]["hidden_size"]
    args.vision_num_attention_heads = llava_args["vision_config"]["num_attention_heads"]
    args.vision_num_layers = llava_args["vision_config"]["num_hidden_layers"]
    # in CLIP-ViT, we use normal multi-head attention
    args.vision_group_query_attention = False
    # so we set vision_num_query_groups to vision_num_attention_heads
    # args.vision_num_query_groups = args.vision_num_attention_heads
    args.vision_kv_channels = args.vision_hidden_size // args.vision_num_attention_heads
    args.vision_ffn_hidden_size = llava_args["vision_config"]["intermediate_size"]
    args.vision_gated_linear_unit = False
    args.vision_swiglu = False  # this is required to turn off gated linear unit
    args.vision_add_position_embedding = True
    args.vision_normalization = "LayerNorm"
    args.vision_add_qkv_bias = True
    args.vision_add_bias_linear = True
    # args.vision_apply_query_key_layer_scaling = True

    # projector config
    args.projector_ffn_hidden_size = llava_args["vision_config"]["intermediate_size"]
    args.projector_hidden_size = llava_args["vision_config"]["intermediate_size"]
    args.projector_add_bias_linear = True
    args.projector_gated_linear_unit = False
    args.projector_swiglu = False

    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True

    args.tokenizer_type = "Llama2Tokenizer"
    args.fp16 = False  # True

    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True

    # args.vocab_size = llava_args["vocab_size"]
    args.vocab_size = llava_args["text_config"]["vocab_size"]

    # args.padded_vocab_size = llava_args["vocab_size"]
    args.padded_vocab_size = llava_args["text_config"]["vocab_size"]

    args.llama = llava_args


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("loader_mcore_llava | Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--load', args.load_dir,
                '--position-embedding-type', args.position_embedding_type,
                ]

    margs = parse_args()

    margs.tensor_model_parallel_size = args.tensor_model_parallel_size
    margs.pipeline_model_parallel_size = args.pipeline_model_parallel_size

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    margs, checkpoint_args = load_args_from_checkpoint(margs, exit_on_missing_checkpoint=True)

    # override some args
    load_llava_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    # Explicitly copy data types from checkpoint.
    margs.fp16 = checkpoint_args.fp16
    margs.bf16 = checkpoint_args.bf16

    # Validate margs.
    margs = validate_args(margs)

    margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"loader_mcore_llava | Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"loader_mcore_llava | Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models
    from scripts.train.llava_model_provider import model_provider
    margs.model_type = ModelType.encoder_or_decoder

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None
    def get_models(count, dtype):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        model_array_len = margs.virtual_pipeline_model_parallel_size
        if model_array_len is None:
            model_array_len = 1
        models = [[] for _ in range(model_array_len)]
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        for rank in range(count):
            mpu.set_tensor_model_parallel_rank(rank)
            if margs.virtual_pipeline_model_parallel_size is not None:
                model_ = []
                for i in range(margs.virtual_pipeline_model_parallel_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(
                        pre_process=pre_process,
                        post_process=post_process
                    ).to(dtype)
                    model_.append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                model_rank = 0
                model_ = [model_provider(add_encoder=True).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            margs.exit_on_missing_checkpoint = True
            load_checkpoint(model_, None, None, strict=False)

            if consumed_train_samples is not None:
                assert(margs.consumed_train_samples == consumed_train_samples), f"{margs.consumed_train_samples} != {consumed_train_samples}"
            else:
                consumed_train_samples = margs.consumed_train_samples
            if consumed_valid_samples is not None:
                assert(margs.consumed_valid_samples == consumed_valid_samples)
            else:
                consumed_valid_samples = margs.consumed_valid_samples
            for vp_rank in range(model_array_len):
                models[vp_rank].append(model_[vp_rank])

            # Print memory usage.
            print_memory_usage("loader", rank, count)

        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("loader_mcore_llava | Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.use_legacy_models = margs.use_legacy_models

    # # Get transformer block (named either 'encoder' or 'decoder').
    # transformer_block_key = get_mcore_transformer_block_key(md.model_type)
    # def get_transformer_block(_model):
    #     return getattr(_model, transformer_block_key)

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(tp_size, md.params_dtype)]
    models = all_models[0][0]

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    print("loader_mcore_llava | Sending metadata...")
    queue.put(md)
    print("loader_mcore_llava | Metadata sent.")

    def queue_put(name, msg):
        print(f"loader_mcore_llava | sending {name}")
        msg["name"] = name
        queue.put(msg, block=True, timeout=None)
        if name.startswith("transformer layer"):
            for key, value in msg.items():
                if key != "name":
                    try:
                        print(f"  {name} {key}: {value.mean().item()}")
                    except Exception as e:
                        print(f"  {name} {key}: {e}")

    # 1. Send Encoders
    # 1.1 Embeddings (conv1, position_embeddings, ln_pre)
    # note: in embeddings, tensor parallelism is not used, so we just send the first model
    vision_conv1_weight = models[0].vision_model.conv1.weight.data
    vision_position_embeddings_weight = models[0].vision_model.position_embeddings.weight.data
    vision_ln_pre_weight = models[0].vision_model.ln_pre.weight.data
    vision_ln_pre_bias = models[0].vision_model.ln_pre.bias.data
    message = {
        "conv1 weight": vision_conv1_weight,
        "position embeddings": vision_position_embeddings_weight,
        "ln pre weight": vision_ln_pre_weight,
        "ln pre bias": vision_ln_pre_bias,
        "class embeddings": models[0].vision_model.class_token
    }
    queue_put("vision embeddings", message)

    # 1.2 Vision Encoders transformer blocks
    total_layer_num = 0
    vision_transformer_layer_num = len(models[0].vision_model.transformer.layers)
    print(f"loader_mcore_llava | vision_transformer_layer_num: {vision_transformer_layer_num}")

    for layer_num in range(vision_transformer_layer_num):
        layer = models[0].vision_model.transformer.layers[layer_num]
        message = {}
        # Get non-parallel tensors from tp_rank 0
        message["input norm weight"] = layer.input_layernorm.weight.data
        message["input norm bias"] = layer.input_layernorm.bias.data
        message["pre norm weight"] = layer.pre_mlp_layernorm.weight.data
        message["pre norm bias"] = layer.pre_mlp_layernorm.bias.data

        # Grab all parallel tensors for this layer
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        for tp_rank, model in enumerate(models):
            layer = model.vision_model.transformer.layers[layer_num]
            qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
            dense_weight.append(layer.self_attention.linear_proj.weight.data)
            mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
            mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)

            qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
            mlp_l0_bias.append(layer.mlp.linear_fc1.bias.data)

        # simple concat of the rest
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
        message["mlp l1 bias"] = layer.mlp.linear_fc2.bias.data
        message["dense bias"] = layer.self_attention.linear_proj.bias.data

        queue_put(f"vision transformer layer {total_layer_num}", message)

        total_layer_num = total_layer_num + 1
    # 1.3 Send final layer norm; no need to send this
    # message = {
    #     "final layer norm weight": models[0].vision_model.transformer.final_layernorm.weight.data,
    #     "final layer norm bias": models[0].vision_model.transformer.final_layernorm.bias.data
    # }
    # queue_put("vision final layer norm", message)


    # 2. Send Adapters
    adapter_linear_fc1_weight = torch.cat(
        [models[tp_rank].vision_projection.encoder.linear_fc1.weight.data for tp_rank in range(tp_size)],
        dim=0
    )
    adapter_linear_fc2_weight = torch.cat(
        [models[tp_rank].vision_projection.encoder.linear_fc2.weight.data for tp_rank in range(tp_size)],
        dim=1
    )
    adapter_linear_fc1_bias = torch.cat(
        [models[tp_rank].vision_projection.encoder.linear_fc1.bias.data for tp_rank in range(tp_size)],
        dim=0
    )
    adapter_linear_fc2_bias = models[0].vision_projection.encoder.linear_fc2.bias.data
    message = {
        "linear fc1 weight": adapter_linear_fc1_weight,
        "linear fc2 weight": adapter_linear_fc2_weight,
        "linear fc1 bias": adapter_linear_fc1_bias,
        "linear fc2 bias": adapter_linear_fc2_bias
    }
    queue_put("adapter", message)

    # 3. Send Decoders
    # Send embeddings
    message = {
        "word embeddings": torch.cat(
            [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim=0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].language_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(models[0].language_model.embedding, 'position_embeddings')

    queue_put("language embeddings", message)

    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            print(f"loader_mcore_llava | model_{pp_rank}_{vp_rank}")
            for layer_num, layer in enumerate(models[0].language_model.decoder.layers):
                message = {}

                # Get non-parallel tensors from tp_rank 0
                message["input norm weight"] = layer.input_layernorm.weight.data
                message["pre mlp norm weight"] = layer.pre_mlp_layernorm.weight.data

                # Grab all parallel tensors for this layer
                qkv_weight = []
                qkv_bias = []
                dense_weight = []
                mlp_l0_weight = []
                mlp_l0_bias = []
                mlp_l1_weight = []
                for tp_rank, model in enumerate(models):
                    layer = model.language_model.decoder.layers[layer_num]
                    qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
                    dense_weight.append(layer.self_attention.linear_proj.weight.data)
                    mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
                    mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)

                # Handle gated linear units
                if md.swiglu:
                    print("loader_mcore_llava | swiglu")
                    # concat all the first halves ('W's) and all the second halves ('V's)
                    for tp_rank in range(tp_size):
                        mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                    message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                # simple concat of the rest
                message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                message["dense weight"] = torch.cat(dense_weight, dim=1)
                message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
                if md.linear_bias:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if md.swiglu:
                        for tp_rank in range(tp_size):
                            mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                        message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                        message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
                    else:
                        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

                queue_put(f"transformer layer {total_layer_num}", message)

                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = {
        "weight": models[0].language_model.decoder.final_layernorm.weight.data,
    }
    if norm_has_bias:
        message["bias"] = models[0].language_model.decoder.final_layernorm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": torch.cat(
                [models[tp_rank].language_model.output_layer.weight.data for tp_rank in range(tp_size)],
                dim=0)
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
