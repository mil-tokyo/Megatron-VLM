# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--saver-transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def save_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import (parse_args, validate_args)
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get(name=None):
        # print(f"waiting for {name}" if name is not None else "waiting for None")
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
            val["name"] = name
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"(saver_megatron) Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()
    print(f"Received model data: {type(md)}")

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
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
                '--save-interval', '1',
                '--save', args.save_dir
                ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()

    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay']

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                if arg == 'decoder_seq_length':
                    print(f"Skipping decoder_seq_length check. Got {value}, expected {getattr(margs, arg)}")
                    continue
                print(f"Megatron Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    validate_args(margs)

    # Use MLM models.
    margs.use_mcore_models = False
    margs.transformer_impl = args.saver_transformer_impl

    set_global_variables(margs, build_tokenizer=False)

    # margs = megatron args
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    print(f"model type: {md.model_type}")

    # Determine how to make our models
    if md.model_type == 'LLaVA':  # ADDED!
        from scripts.train.llava_model_provider import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    def get_models(count, dtype, add_encoder: bool):
        models = [model_provider(add_encoder).to(dtype) for _ in range(count)]
        return models

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Make models for first pipeline stage and set vision model
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, add_encoder=True)

    # vision model
    # ------------
    # vision model pre
    vision_model_pre_msg = queue_get("vision model pre")
    # message = {
    #         "conv1": model.vision_model.conv1.weight.data,
    #         "position embeddings": model.vision_model.position_embeddings.weight.data,
    #         "ln_pre": model.vision_model.ln_pre.weight.data
    #     }
    conv1 = vision_model_pre_msg.pop("conv1")
    position_embeddings = vision_model_pre_msg.pop("position embeddings")
    is_siglip = False
    if "class token" in vision_model_pre_msg:
        # clip
        class_token = vision_model_pre_msg.pop("class token")
        ln_pre_weight = vision_model_pre_msg.pop("ln_pre weight")
        ln_pre_bias = vision_model_pre_msg.pop("ln_pre bias")
        conv1_bias = None
    else:
        is_siglip = True
        conv1_bias = vision_model_pre_msg.pop("conv1 bias")
        print("\nclass token not found in vision model pre message (maybe this vision model is siglip")
    check_message(vision_model_pre_msg)

    # put conv1, position_embeddings, ln_pre into model
    # these are the same for all tp_ranks
    for tp_rank, model in enumerate(models):
        model.vision_model.conv1.weight.data.copy_(conv1)
        if conv1_bias is not None:
            model.vision_model.conv1.bias.data.copy_(conv1_bias)
        model.vision_model.position_embeddings.weight.data.copy_(position_embeddings)
        if not is_siglip:
            model.vision_model.class_token.data.copy_(class_token)
            model.vision_model.ln_pre.weight.data.copy_(ln_pre_weight)
            model.vision_model.ln_pre.bias.data.copy_(ln_pre_bias)

    # vision model transformer layers
    # ------------------------------
    total_layer_num = 0
    print("start saving vision transformer layers")

    print(f"size of vision transformer layers: {len(models[0].vision_model.transformer.layers)}")
    for layer in range(len(models[0].vision_model.transformer.layers)):
        msg = queue_get(f"vision transformer layer {total_layer_num}")
        # e.g.
        # vision transformer layer 0
        #     self_attention linear_proj weight
        #     self_attention linear_qkv weight
        #     input_layernorm weight
        #     input_layernorm bias
        #     mlp linear_fc1 weight
        #     mlp linear_fc2 weight
        #     pre_mlp_layernorm weight
        #     pre_mlp_layernorm bias

        # Split up the parallel tensors
        qkv_weight = torch.chunk(msg.pop("self_attention linear_qkv weight"), args.target_tensor_parallel_size, dim=0)
        qkv_bias = torch.chunk(msg.pop("self_attention linear_qkv bias"), args.target_tensor_parallel_size, dim=0)

        dense_weight = torch.chunk(msg.pop("self_attention linear_proj weight"), args.target_tensor_parallel_size, dim=1)
        # dense_bias = torch.chunk(msg.pop("self_attention linear_proj bias"), args.target_tensor_parallel_size, dim=0)
        dense_bias = msg.pop("self_attention linear_proj bias")

        mlp_l1_weight = torch.chunk(msg.pop("mlp linear_fc1 weight"), args.target_tensor_parallel_size, dim=0)
        mlp_l1_bias = torch.chunk(msg.pop("mlp linear_fc1 bias"), args.target_tensor_parallel_size, dim=0)

        mlp_l2_weight = torch.chunk(msg.pop("mlp linear_fc2 weight"), args.target_tensor_parallel_size, dim=1)
        # mlp_l2_bias = torch.chunk(msg.pop("mlp linear_fc2 bias"), args.target_tensor_parallel_size, dim=0)
        mlp_l2_bias = msg.pop("mlp linear_fc2 bias")

        input_norm_weight = msg.pop("input_layernorm weight")
        input_norm_bias = msg.pop("input_layernorm bias")
        pre_layer_norm_weight = msg.pop("pre_mlp_layernorm weight")
        pre_layer_norm_bias = msg.pop("pre_mlp_layernorm bias")

        # Save them to the model
        for tp_rank in range(args.target_tensor_parallel_size):
            l = models[tp_rank].vision_model.transformer.layers[layer]
            l.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
            l.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])

            l.self_attention.linear_proj.weight.data.copy_(dense_weight[tp_rank])
            l.self_attention.linear_proj.bias.data.copy_(dense_bias)

            l.input_layernorm.weight.data.copy_(input_norm_weight)
            l.input_layernorm.bias.data.copy_(input_norm_bias)
            l.pre_mlp_layernorm.weight.data.copy_(pre_layer_norm_weight)
            l.pre_mlp_layernorm.bias.data.copy_(pre_layer_norm_bias)

            l.mlp.linear_fc1.weight.data.copy_(mlp_l1_weight[tp_rank])
            l.mlp.linear_fc1.bias.data.copy_(mlp_l1_bias[tp_rank])
            l.mlp.linear_fc2.weight.data.copy_(mlp_l2_weight[tp_rank])
            l.mlp.linear_fc2.bias.data.copy_(mlp_l2_bias)

        total_layer_num = total_layer_num + 1
        # check_message(msg)

        if msg != "done" and msg["name"] == "lm head":
            if not hasattr(models[0], 'lm_head'):
                print("ERROR: got an lm head, but model does not have one")
                exit(1)
            print("received lm head")
            lm_head_dense_weight = msg.pop("dense weight")
            lm_head_dense_bias = msg.pop("dense bias")
            lm_head_norm_weight = msg.pop("norm weight")
            if md.norm_has_bias:
                lm_head_norm_bias = msg.pop("norm bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                models[tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                if md.norm_has_bias:
                    models[tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
            check_message(msg)
            msg = queue_get()

        if msg != "done" and msg["name"] == "binary head":
            if not hasattr(models[0], 'binary_head'):
                print("ERROR: got a binary head, but model does not have one")
                exit(1)
            print("received binary head")
            binary_head_weight = msg.pop("weight")
            binary_head_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
            check_message(msg)
            msg = queue_get()

        # msg is not "done" and msg has more keys than just "name"
        if msg != "done" and len(msg.keys()) > 1:
            print(f"ERROR: got some more data but was expecting to be done: {msg}")

    # vision final norm
    # -----------------
    # msg = queue_get("vision model post")
    # final_norm_weight = msg.pop("weight")
    # final_norm_bias = msg.pop("bias")
    # for tp_rank in range(args.target_tensor_parallel_size):
    #     models[tp_rank].vision_model.transformer.final_layernorm.weight.data.copy_(final_norm_weight)
    #     models[tp_rank].vision_model.transformer.final_layernorm.bias.data.copy_(final_norm_bias)

    # vision projection MLP
    # ---------------------
    vision_projection_msg = queue_get("vision projection MLP")
    vision_projection_l1_weight = torch.chunk(vision_projection_msg.pop("projection fc1"), args.target_tensor_parallel_size, dim=0)
    vision_projection_l2_weight = torch.chunk(vision_projection_msg.pop("projection fc2"), args.target_tensor_parallel_size, dim=1)
    # vision_projection_l1_bias = vision_projection_msg.pop("projection fc1 bias")
    vision_projection_l1_bias = torch.chunk(vision_projection_msg.pop("projection fc1 bias"), args.target_tensor_parallel_size, dim=0)
    vision_projection_l2_bias = vision_projection_msg.pop("projection fc2 bias")
    check_message(vision_projection_msg)

    for tp_rank in range(args.target_tensor_parallel_size):
        models[tp_rank].vision_projection.encoder.linear_fc1.weight.data.copy_(vision_projection_l1_weight[tp_rank])
        models[tp_rank].vision_projection.encoder.linear_fc2.weight.data.copy_(vision_projection_l2_weight[tp_rank])
        # models[tp_rank].vision_projection.encoder.linear_fc1.bias.data.copy_(vision_projection_l1_bias)
        models[tp_rank].vision_projection.encoder.linear_fc1.bias.data.copy_(vision_projection_l1_bias[tp_rank])
        models[tp_rank].vision_projection.encoder.linear_fc2.bias.data.copy_(vision_projection_l2_bias)

    print("\nstart saving language model")
    # Embeddings
    # -----------
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # Deal with padding
    print(f"true_vocab_size: {md.true_vocab_size}")
    if md.true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

        # Cut out extra padding we don't need
        if orig_vocab_size > margs.padded_vocab_size:
            full_word_embed = orig_word_embed[0:margs.padded_vocab_size,:]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < margs.padded_vocab_size:
            padding_size = margs.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        margs.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    # Make models for first pipeline stage and fill in embeddings
    # mpu.set_pipeline_model_parallel_rank(0)
    # post_process = args.target_pipeline_parallel_size == 1
    # models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)
    for tp_rank, model in enumerate(models):
        # model.language_model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        model.language_model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            # model.language_model.embedding.position_embeddings.weight.data.copy_(pos_embed)
            model.language_model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            # assert not hasattr(model.language_model.embedding, "position_embeddings")
            assert not hasattr(model.language_model.embedding, "position_embeddings")

    # Transformer layers
    # ------------------
    total_layer_num = 0
    print("start saving transformer layers")
    # decoder_pp_size = args.target_pipeline_parallel_size - encoder_pp_size
    decoder_pp_size = args.target_pipeline_parallel_size
    print(f"decoder_pp_size: {decoder_pp_size}")
    for pp_rank in range(decoder_pp_size):
        print(f"\nProcessing pipeline rank {pp_rank}")
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            print(f"Creating new models for pipeline rank {pp_rank}")
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == decoder_pp_size - 1
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, add_encoder=False)
        else:
            print(f"Using existing models for pipeline rank {pp_rank}")

        # for layer in range(len(models[0].language_model.encoder.layers)):
        for layer in range(len(models[0].language_model.decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")  #TODO: NEED REVIEW
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")  #TODO: NEED REVIEW
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].language_model.decoder.layers[layer]
                l.input_layernorm.weight.data.copy_(input_norm_weight)  #TODO: NEED REVIEW
                if md.norm_has_bias:
                    l.input_norm.bias.data.copy_(input_norm_bias)
                l.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
                l.self_attention.linear_proj.weight.data.copy_(dense_weight[tp_rank])
                l.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)  #TODO: NEED REVIEW
                if md.norm_has_bias:
                    l.post_attention_norm.bias.data.copy_(post_norm_bias)
                l.mlp.linear_fc1.weight.data.copy_(mlp_l0_weight[tp_rank])
                l.mlp.linear_fc2.weight.data.copy_(mlp_l1_weight[tp_rank])
                if md.linear_bias:
                    l.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
                    l.self_attention.linear_proj.bias.data.copy_(dense_bias)
                    l.mlp.linear_fc1.bias.data.copy_(mlp_l0_bias[tp_rank])
                    l.mlp.linear_fc2.bias.data.copy_(mlp_l1_bias[tp_rank])

            total_layer_num = total_layer_num + 1
            check_message(msg)

        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                # models[tp_rank].language_model.encoder.final_norm.weight.data.copy_(final_norm_weight)
                models[tp_rank].language_model.decoder.final_layernorm.weight.data.copy_(final_norm_weight)
                if md.norm_has_bias:
                    # models[tp_rank].language_model.encoder.final_norm.bias.data.copy_(final_norm_bias)
                    models[tp_rank].language_model.decoder.final_layernorm.bias.data.copy_(final_norm_bias)
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].language_model.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                # if not hasattr(models[0].language_model, 'output_layer'):
                if not hasattr(models[0].language_model, 'output_layer'):
                    print("ERROR: got an output layer, but model does not have one")
                    exit(1)
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                for tp_rank in range(args.target_tensor_parallel_size):
                    # models[tp_rank].language_model.output_layer.weight.data.copy_(output_layer_weight[tp_rank])
                    models[tp_rank].language_model.output_layer.weight.data.copy_(output_layer_weight[tp_rank])
                del output_layer_weight
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                # if not hasattr(models[0].language_model, 'pooler'):
                if not hasattr(models[0].language_model, 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    # models[tp_rank].language_model.pooler.dense.weight.data.copy_(pooler_weight)
                    # models[tp_rank].language_model.pooler.dense.bias.data.copy_(pooler_bias)
                    models[tp_rank].language_model.pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].language_model.pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                    if md.norm_has_bias:
                        models[tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        print("done saving transformer layers")
        print("\nsaving checkpoint...")
        for tp_rank in range(args.target_tensor_parallel_size):
            # decoder_pp_rank = pp_rank + encoder_pp_size
            decoder_pp_rank = pp_rank
            print(f"Saving model (tp {tp_rank}, pp {decoder_pp_rank})")
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_pipeline_parallel = True if args.target_pipeline_parallel_size > 1 else None
            print(f"save_pipeline_parallel: {save_pipeline_parallel}")
            save_checkpoint(
                md.iteration, [models[tp_rank]], None, None,
                num_floating_point_operations_so_far=0,
                pipeline_parallel=save_pipeline_parallel,
                pipeline_rank=decoder_pp_rank,
            )
    print("Done!")
