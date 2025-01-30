# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
from copy import deepcopy

import torch
import transformers
from tqdm import tqdm
import types

from transformers import AutoConfig
from transformers.models.clip.modeling_clip import CLIPAttention
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipSdpaAttention

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.convert.modeling_llava import LlavaForConditionalGeneration


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


def load_args_from_checkpoint(args):
    print("Loading args from checkpoint... (loader_llava_hf.py)")

    # Read Llama args.
    load_args_from_config_json = False
    print(f"Loading args from {args.load}")
    if args.config_json_path is None:
        llava_args_path = os.path.join(args.load, "config.json")
    else:
        llava_args_path = args.config_json_path
    if os.path.exists(llava_args_path):
        load_args_from_config_json = True
    else:
        print(f"config.json not found in the checkpoint directory: {args.load}")

    if load_args_from_config_json:
        print(f"Loading args from {llava_args_path}")
        with open(llava_args_path) as f:
            llava_args = json.load(f)
    else:
        raise ValueError("You must provide a config.json file to load the model.")

    # load text config
    text_model_name_or_path = llava_args["text_config"]["_name_or_path"]
    print(f"Loading text model config from {text_model_name_or_path}")
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
    if llava_args["vision_config"]["model_type"] == "clip_vision_model":
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
        args.vision_model_subtype = "clip"
        # args.vision_apply_query_key_layer_scaling = True
    elif llava_args["vision_config"]["model_type"] == "siglip_vision_model":
        # set siglip config
        args.vision_num_layers = llava_args["vision_config"]["num_hidden_layers"]
        args.vision_num_attention_heads = llava_args["vision_config"]["num_attention_heads"]
        args.vision_add_bias_linear = True
        args.vision_add_qkv_bias = True
        args.vision_hidden_size = llava_args["vision_config"]["hidden_size"]
        args.vision_hidden_dropout = 0.0
        args.vision_attention_dropout = 0.0
        args.vision_ffn_hidden_size = llava_args["vision_config"]["intermediate_size"]
        args.vision_gated_linear_unit = False
        args.vision_swiglu = False  # this is required to turn off gated linear unit
        args.vision_kv_channels = args.vision_hidden_size // args.vision_num_attention_heads
        args.vision_num_query_groups = 16  # TODO: remove hardcoding
        args.vision_layernorm_zero_centered_gamma = False
        # args.vision_apply_query_key_layer_scaling = False
        args.vision_bias_activation_fusion = False
        args.vision_bias_dropout_fusion = False
        args.vision_attention_softmax_in_fp32 = True
        args.vision_normalization = "LayerNorm"
        args.vision_apply_rope_fusion = False
        args.vision_qk_layernorm = False
        args.vision_layernorm_epsilon = 1e-6
        args.vision_model_subtype = "siglip"
        args.add_class_token = 0

    # projector config
    if "projector_hidden_size" in llava_args:
        projector_hidden_size = llava_args["projector_hidden_size"]
    else:
        projector_hidden_size = llava_args["text_config"].get("hidden_size", 4096)
    print(f"projector_hidden_size: {projector_hidden_size}")
    args.projector_ffn_hidden_size = projector_hidden_size
    args.projector_hidden_size = llava_args["text_config"].get("hidden_size", 4096)
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

    args.vocab_size = llava_args["text_config"]["vocab_size"]

    args.padded_vocab_size = llava_args["text_config"]["vocab_size"]

    args.llama = llava_args


def set_vision_module_state(args, model, hf_model):
    '''Set vision module params.'''

    # Get vision module.
    vision_module = model.vision_model
    # megatron's vision model is like this:
    # (vision_model): CLIPViTModel(
    #     (conv1): Conv2d(3, 4096, kernel_size=(14, 14), stride=(14, 14), bias=False)
    #     (position_embeddings): Embedding(577, 4096)
    #     (ln_pre): RMSNorm()
    #     (transformer): TransformerBlock(
    #       (layers): ModuleList(
    #         (0-31): 32 x TransformerLayer(
    #           (input_layernorm): IdentityOp()
    #           (self_attention): SelfAttention(
    #             (core_attention): TEDotProductAttention(
    #               (flash_attention): FlashAttention()
    #               (fused_attention): FusedAttention()
    #               (unfused_attention): UnfusedDotProductAttention(
    #                 (scale_mask_softmax): FusedScaleMaskSoftmax()
    #                 (attention_dropout): Dropout(p=0.1, inplace=False)
    #               )
    #             )
    #             (linear_proj): TERowParallelLinear()
    #             (linear_qkv): TELayerNormColumnParallelLinear()
    #             (q_layernorm): IdentityOp()
    #             (k_layernorm): IdentityOp()
    #           )
    #           (pre_cross_attn_layernorm): IdentityOp()
    #           (cross_attention): IdentityOp()
    #           (cross_attn_bda): IdentityFuncOp()
    #           (pre_mlp_layernorm): IdentityOp()
    #           (mlp): MLP(
    #             (linear_fc1): TEColumnParallelLinear()
    #             (linear_fc2): TERowParallelLinear()
    #           )
    #         )
    #       )
    #       (final_layernorm): RMSNorm()
    #     )
    #   )

    hf_vision_module = hf_model.vision_tower
    # hf_model's vision model is like this:
    # (vision_tower): CLIPVisionModel(
    #     (vision_model): CLIPVisionTransformer(
    #       (embeddings): CLIPVisionEmbeddings(
    #         (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
    #         (position_embedding): Embedding(577, 1024)
    #       )
    #       (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    #       (encoder): CLIPEncoder(
    #         (layers): ModuleList(
    #           (0-23): 24 x CLIPEncoderLayer(
    #             (self_attn): CLIPAttention(
    #               (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
    #               (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
    #               (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
    #               (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
    #             )
    #             (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    #             (mlp): CLIPMLP(
    #               (activation_fn): QuickGELUActivation()
    #               (fc1): Linear(in_features=1024, out_features=4096, bias=True)
    #               (fc2): Linear(in_features=4096, out_features=1024, bias=True)
    #             )
    #             (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    #           )
    #         )
    #       )
    #       (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    #     )
    #   )

    # Set vision module state.
    if hasattr(vision_module, "class_token"):
        vision_module_type = "clip"
    else:
        vision_module_type = "siglip"
    print(f"vision module type: {vision_module_type} (from args.vision_model_subtype: {args.vision_model_subtype})")

    print("setting vision module state...")
    # vision embedding
    vision_module.conv1.weight.data.copy_(hf_vision_module.vision_model.embeddings.patch_embedding.weight)
    if vision_module_type == "siglip":
        vision_module.conv1.bias.data.copy_(hf_vision_module.vision_model.embeddings.patch_embedding.bias)

    vision_module.position_embeddings.weight.data.copy_(hf_vision_module.vision_model.embeddings.position_embedding.weight)
    # class token from vision_tower.vision_model.embeddings.class_embedding
    if vision_module_type == "clip":
        vision_module.class_token.data.copy_(hf_vision_module.vision_model.embeddings.class_embedding)

    # pre layer norm
    if args.vision_model_subtype == "clip":
        vision_module.ln_pre.weight.data.copy_(hf_vision_module.vision_model.pre_layrnorm.weight)
        vision_module.ln_pre.bias.data.copy_(hf_vision_module.vision_model.pre_layrnorm.bias)

    # set transformer layers
    print("setting transformer layers...")
    for layer_idx in range(args.vision_num_layers):
        set_vision_layer_state(args, vision_module, hf_vision_module, layer_idx)

    # set final layer norm; actually this is not used in LLaVA
    # vision_module.transformer.final_layernorm.weight.data.copy_(hf_vision_module.vision_model.post_layernorm.weight)
    # vision_module.transformer.final_layernorm.bias.data.copy_(hf_vision_module.vision_model.post_layernorm.bias)


def set_projector_state(args, model, hf_model):
    '''Set projector params.'''

    # Get projector.
    projector = model.vision_projection
    hf_projector = hf_model.multi_modal_projector

    # Set projector state.
    print("setting projector state...")
    projector.encoder.linear_fc1.weight.data.copy_(hf_projector.linear_1.weight)
    projector.encoder.linear_fc2.weight.data.copy_(hf_projector.linear_2.weight)
    projector.encoder.linear_fc1.bias.data.copy_(hf_projector.linear_1.bias)
    projector.encoder.linear_fc2.bias.data.copy_(hf_projector.linear_2.bias)


def set_vision_layer_state(args, vision_module, hf_vision_module, layer_idx):
    '''Set transformer layer params.'''
    # get vision args
    args = deepcopy(args)

    vision_args = {k: v for k, v in vars(args).items() if k.startswith("vision_")}
    # overwrite args with vision args
    for k, v in vision_args.items():
        args_key_without_vision = k.replace("vision_", "")
        setattr(args, args_key_without_vision, v)
        # remove vision args from args
        delattr(args, k)

    # Get vision layer & state.
    megatron_layer_num = len(vision_module.transformer.layers)
    hf_layer_num = len(hf_vision_module.vision_model.encoder.layers)
    if layer_idx == megatron_layer_num:
        print(f"ViT final layer ({layer_idx}) is not used in LLaVA")
        return

    layer = vision_module.transformer.layers[layer_idx]
    hf_layer = hf_vision_module.vision_model.encoder.layers[layer_idx]

    # Set self-attention params.
    set_attn_state(args, layer, hf_layer)

    # Set MLP params.
    set_mlp_state_clip(args, layer, hf_layer)


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    print("setting preprocess state...")
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    print("setting postprocess state...")
    # model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    # model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    # q_proj.weight: (1024, 1024) -> (16, 128, 512)
    reshaped_q = hf_attn.q_proj.weight.reshape((nh, dim*nh//ng, -1))
    reshaped_k = hf_attn.k_proj.weight.reshape((nh, dim, -1))
    reshaped_v = hf_attn.v_proj.weight.reshape((nh, dim, -1))
    # print(f"nh: {nh}, dim: {dim}, ng: {ng}")
    # print(f"hf q_proj weight size: {hf_attn.q_proj.weight.size()} -> {reshaped_q.size()}")
    # print(f"hf k_proj weight size: {hf_attn.k_proj.weight.size()} -> {reshaped_k.size()}")
    # print(f"hf v_proj weight size: {hf_attn.v_proj.weight.size()} -> {reshaped_v.size()}")

    # llava7b (tp2, pp1) (vision)
    # nh: 16, dim: 64, ng: 16
    # hf q_proj weight size: torch.Size([1024, 1024]) -> torch.Size([16, 64, 1024])
    # hf k_proj weight size: torch.Size([1024, 1024]) -> torch.Size([16, 64, 1024])
    # hf v_proj weight size: torch.Size([1024, 1024]) -> torch.Size([16, 64, 1024])
    # concat qkv size: torch.Size([16, 192, 1024])
    # attn.linear_qkv.weight.size(): torch.Size([3072, 1024])

    # llava13b (tp4, pp1) (text)
    # nh: 40, dim: 128, ng: 40
    # hf q_proj weight size: torch.Size([5120, 5120]) -> torch.Size([40, 128, 5120])
    # hf k_proj weight size: torch.Size([5120, 5120]) -> torch.Size([40, 128, 5120])
    # hf v_proj weight size: torch.Size([5120, 5120]) -> torch.Size([40, 128, 5120])
    # concat qkv size: torch.Size([40, 384, 5120])
    # attn.linear_qkv.weight.size(): torch.Size([15360, 5120])

    # llava1b (tp1, pp1) (text)
    # nh: 32, dim: 64, ng: 4
    # hf q_proj weight size: torch.Size([2048, 2048]) -> torch.Size([32, 512, 256])
    # hf k_proj weight size: torch.Size([256, 2048]) -> torch.Size([32, 64, 256])
    # hf v_proj weight size: torch.Size([256, 2048]) -> torch.Size([32, 64, 256])
    # concat qkv size: torch.Size([32, 640, 256])
    # attn.linear_qkv.weight.size(): torch.Size([2560, 2048])
    concat_qkv = torch.cat([reshaped_q, reshaped_k, reshaped_v], dim=1)
    # print(f"concat qkv size: {concat_qkv.size()}")
    attn.linear_qkv.weight.data.copy_(concat_qkv.reshape((-1, args.hidden_size)))  # (3072, 1024) <- this has to be same as linear_qkv
    # print(f"attn.linear_qkv.weight.size(): {attn.linear_qkv.weight.size()}")

    if isinstance(hf_attn, CLIPAttention) or isinstance(hf_attn, SiglipSdpaAttention):
        # print(f"ng: {ng}, dim: {dim}, nh: {nh}, dim*nh//ng: {dim * nh // ng}")
        # print(f"reshaped q_proj size: {hf_attn.q_proj.weight.reshape((ng, dim * nh // ng, -1)).size()}")
        # print(f"reshaped k_proj size: {hf_attn.k_proj.weight.reshape((ng, dim * nh // ng, -1)).size()}")
        # print(f"reshaped v_proj size: {hf_attn.v_proj.weight.reshape((ng, dim * nh // ng, -1)).size()}")
        # print(f"attn.linear_qkv.weight.size(): {attn.linear_qkv.weight.size()}")
        #
        # print(f"attn.linear_qkv.bias.size(): {attn.linear_qkv.bias.size()}")
        # print(f"hf_attn.q_proj.bias.size(): {hf_attn.q_proj.bias.size()}")
        # print(f"hf_attn.k_proj.bias.size(): {hf_attn.k_proj.bias.size()}")
        # print(f"hf_attn.v_proj.bias.size(): {hf_attn.v_proj.bias.size()}")
        attn.linear_qkv.bias.data.copy_(torch.cat([
            hf_attn.q_proj.bias.reshape((ng, -1)),
            hf_attn.k_proj.bias.reshape((ng, -1)),
            hf_attn.v_proj.bias.reshape((ng, -1)),
        ], dim=1).reshape(-1))
        attn.linear_proj.weight.data.copy_(hf_attn.out_proj.weight)
        attn.linear_proj.bias.data.copy_(hf_attn.out_proj.bias)
    else:
        if hasattr(hf_attn, "out_proj"):
            attn.linear_proj.weight.data.copy_(hf_attn.out_proj.weight)
        else:
            attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    # ADD!
    # Copy weights and biases of layer norm.
    # print("copying layer norm weights...")
    if isinstance(hf_attn, CLIPAttention) or isinstance(hf_layer, SiglipEncoderLayer):
        # weight
        layer.input_layernorm.weight.data.copy_(hf_layer.layer_norm1.weight)
        layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.layer_norm2.weight)
        if hasattr(hf_layer.layer_norm1, 'bias'):
            # bias
            layer.input_layernorm.bias.data.copy_(hf_layer.layer_norm1.bias)
            layer.pre_mlp_layernorm.bias.data.copy_(hf_layer.layer_norm2.bias)
    else:
        # weight
        layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
        layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
        if hasattr(hf_layer.input_layernorm, 'bias'):
            # bias
            layer.input_layernorm.bias.data.copy_(hf_layer.input_layernorm.bias)
            layer.pre_mlp_layernorm.bias.data.copy_(hf_layer.post_attention_layernorm.bias)


def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    # mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
    mlp.linear_fc1.weight.data.copy_(torch.cat([
        hf_mlp.gate_proj.weight,
        hf_mlp.up_proj.weight,
    ], dim=0))
    # mlp.dense_4h_to_h.weight.data.copy_(hf_mlp.down_proj.weight)
    mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)


def set_mlp_state_clip(args, layer, hf_layer) -> None:
    """
    Set MLP params for CLIP model.
    Args:
        args: (argparse.Namespace) Arguments for the model.
        layer: Megatron model layer
        hf_layer: Huuggingface transformer's model layer
    """
    mlp = layer.mlp
    hf_mlp = hf_layer.mlp
    # hf_layer.mlp:
    # CLIPMLP(
    #   (activation_fn): QuickGELUActivation()
    #   (fc1): Linear(in_features=1024, out_features=4096, bias=True)
    #   (fc2): Linear(in_features=4096, out_features=1024, bias=True)
    # )

    # layer.mlp:
    # MLP(
    #   (linear_fc1): TEColumnParallelLinear(in_features=1024, out_features=22016, bias=True)
    #   (linear_fc2): TERowParallelLinear(in_features=11008, out_features=1024, bias=True)
    # )

    # copy weights and biases
    mlp.linear_fc1.weight.data.copy_(hf_mlp.fc1.weight)
    mlp.linear_fc2.weight.data.copy_(hf_mlp.fc2.weight)
    mlp.linear_fc1.bias.data.copy_(hf_mlp.fc1.bias)
    mlp.linear_fc2.bias.data.copy_(hf_mlp.fc2.bias)


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    # layer = model.language_model.encoder.layers[layer_idx]
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    # layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    # layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    # layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(args):
    '''Set model params.'''


    # Load Huggingface model.
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        args.load, device_map="cpu",
        torch_dtype=torch.float32
    )

    # Init Megatron model.
    args.fp16 = False
    print("initializing model...")
    from scripts.train.llava_model_provider import model_provider
    model = model_provider(True).to(args.params_dtype)
    print(f"\n\nmegatron model: \n{model}")

    # Set model state.
    print("setting model state...")

    # save model architecture to be used in the future
    with open(f"{args.save_dir}/model_megatron_llava.txt", "w") as f:
        f.write(str(model))
    with open(f"{args.save_dir}/model_hf_llava.txt", "w") as f:
        f.write(str(hf_model))

    # set vision module state
    print("setting vision module state...")
    set_vision_module_state(args, model, hf_model)

    print("setting language module state...")
    set_preprocess_state(args, model.language_model, hf_model.language_model)
    set_postprocess_state(args, model.language_model, hf_model.language_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model.language_model, hf_model.language_model, layer_idx)

    set_projector_state(args, model, hf_model)

    return model


def _load_checkpoint(queue, args):
    print("loading checkpoint...")

    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
        from scripts.train.llava_model_provider import model_provider
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
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
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    margs.save_dir = args.save_dir
    # added
    # margs.use_dist_ckpt = args.use_dist_ckpt
    # print(f"overwrite margs use_dist_ckpt: {margs.use_dist_ckpt}")
    print("loading args from checkpoint...")
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    print("validating args...")
    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        # print(f"checking for arg {arg_name}")
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
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
    check_for_arg('params_dtype', default='fp32')
    check_for_arg('swiglu', False)

    print(f"args params_dtype: {margs.params_dtype}")
    margs.params_dtype = torch.float32
    args.params_dtype = torch.float32
    # margs.params_dtype = torch.bfloat16
    # args.params_dtype = torch.bfloat16
    print(f"overwrite params_dtype: {margs.params_dtype}")

    # Determine how to make our models.
    assert args.model_type == 'LLaVA', f"Model type must be LLaVA, not {args.model_type}."
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    print("setting global variables...")
    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    print("setting metadata...")
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
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None  # skips padding in saver
    md.make_vocab_size_divisible_by = 128
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    # Get first pipe stage.
    print("get first pipe stage...")
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        # print(f"sending {name}")
        # save the name of the message to txt file like:
        # name
        #    message key1
        #    message key2
        #    ...
        msg["name"] = name
        print(f"queue put: {name}")
        queue.put(msg)

    # Send vision model.
    print("sending vision model...")
    message = {
        "conv1": model.vision_model.conv1.weight.data,
        "position embeddings": model.vision_model.position_embeddings.weight.data,
    }
    if hasattr(model.vision_model, "ln_pre") and model.vision_model.ln_pre is not None:
        # clip only
        # siglip doesn't have this
        message["class token"] = model.vision_model.class_token.data
        message["ln_pre weight"] = model.vision_model.ln_pre.weight.data
        message["ln_pre bias"] = model.vision_model.ln_pre.bias.data
    else:
        # siglip only
        message ["conv1 bias"] = model.vision_model.conv1.bias.data
    queue_put("vision model pre", message)

    # send vision transformer layers
    print(f"\nsending {margs.vision_num_layers} vision transformer layers...")
    for layer_num in range(margs.vision_num_layers):
        # print(f"processing vision layer {layer_num}...")
        message = {}
        # Get non-parallel tensors from tp_rank 0.
        if layer_num == margs.vision_num_layers - 1:
            continue
        layer = model.vision_model.transformer.layers[layer_num]

        # layers to send:
        # - self_attention.linear_proj
        # - self_attention.linear_qkv
        # - input_layernorm
        # - mlp.linear_fc1
        # - mlp.linear_fc2
        # - pre_mlp_layernorm
        layer_state_dict_keys = layer.state_dict().keys()
        layer_to_send = [_l for _l in layer_state_dict_keys if _l.endswith(".weight") or _l.endswith(".bias")]

        # Grab all parallel tensors for this layer.
        for each_layer_to_send in layer_to_send:
            message_layer_to_send = each_layer_to_send.replace(".", " ")
            message[message_layer_to_send] = layer.state_dict()[each_layer_to_send]
        queue_put(f"vision transformer layer {layer_num}", message)

    # send vision_projection MLP
    message = {
        "projection fc1": model.vision_projection.encoder.linear_fc1.weight.data,
        "projection fc2": model.vision_projection.encoder.linear_fc2.weight.data,
        "projection fc1 bias": model.vision_projection.encoder.linear_fc1.bias.data,
        "projection fc2 bias": model.vision_projection.encoder.linear_fc2.bias.data
    }
    queue_put("vision projection MLP", message)

    # send language model
    print("sending embeddings...")
    message = {
        "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    }
    print(f"position embedding type: {md.position_embedding_type}")
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.language_model.embedding, 'position_embeddings'), \
            "Position embeddings should not be present."

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        # print(f"processing language layer {layer_num}...")
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.decoder.layers[layer_num]
        message["input norm weight"] = layer.input_layernorm.weight.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
            message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []

        layer = model.language_model.decoder.layers[layer_num]

        qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
        dense_weight.append(layer.self_attention.linear_proj.weight.data)
        mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
        mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)

        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)

        # Handle gated linear units.
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # Simple concat of the rest.
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

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        # "weight": model.language_model.encoder.final_norm.weight.data,
        "weight": model.language_model.decoder.final_layernorm.weight.data
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            # "weight": model.language_model.output_layer.weight.data
            "weight": model.language_model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    # # clear "scripts/miscs/queue_put_llava.txt"
    # with open("scripts/miscs/queue_put_llava.txt", "w") as f:
    #     f.write("")
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
