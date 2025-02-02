# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
import sys
import os
import torch
import torch.multiprocessing as mp
import transformers
from transformers import AutoTokenizer, AutoConfig
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.convert.modeling_llava import LlavaForConditionalGeneration
from scripts.convert.configuration_llava import LlavaConfig


print("saver_llava3_hf | Launching saver_llava3_hf.py")


def add_arguments(parser):
    group = parser.add_argument_group(title='llava_hf saver.')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='Huggingface tokenizer path. eg. /models/llava-hf')
    group.add_argument('--save-dtype', type=str, default='bfloat16')


@contextmanager
def suspend_nn_inits():
    """
    create context manager for loading without init
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None  # noqa: E731
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_   # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip   # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def save_checkpoint(queue: mp.Queue, args):
    print("saver_llava3_hf | start save_checkpoint")
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 os.path.pardir,
                                                 os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get(block=True, timeout=None)
        if val == "exit":
            print("saver_llava3_hf | Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'saver_llava3_hf | Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"saver_llava3_hf | received {name}")

        if name is not None and name.startswith("transformer layer"):
            for key, value in val.items():
                if key != "name":
                    try:
                        print(f"  {name} {key}: {value.mean().item()}")
                    except Exception as e:
                        print(f"  {name} {key}: {e}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"saver_llava3_hf | Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print("saver_llava3_hf | Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    print("saver_llava3_hf | got checkpoint args")
    assert md.model_type == 'LLaVA', f"Model type {md.model_type} not supported"
    mag_conf = md.checkpoint_args

    if args.save_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.save_dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    llava_conf = LlavaConfig.from_pretrained(f"{args.load_dir}/config.json")
    text_conf = AutoConfig.from_pretrained(llava_conf.text_config._name_or_path)

    state_dict = {}

    def set_hf_param(name, tensor: torch.Tensor, bias=False):
        if bias:
            bias_name = f'{name}.bias'
            state_dict[bias_name] = tensor.to(torch.bfloat16)
        else:
            weight_name = f'{name}.weight'
            state_dict[weight_name] = tensor.to(torch.bfloat16)

    # 1. Encoder
    # 1.1 Vision Embeddings
    message = queue_get("vision embeddings")
    set_hf_param('model.vision_tower.vision_model.embeddings.patch_embedding', message["conv1 weight"])
    set_hf_param('model.vision_tower.vision_model.embeddings.position_embedding', message["position embeddings"])
    set_hf_param('model.vision_tower.vision_model.pre_layrnorm', message["ln pre weight"])
    set_hf_param('model.vision_tower.vision_model.pre_layrnorm', message["ln pre bias"], bias=True)
    state_dict['model.vision_tower.vision_model.embeddings.class_embedding'] = message["class embeddings"].to(torch.bfloat16).view(-1)

    # 1.2 Encoder Transformer Blocks
    # vision_num_key_value_heads: 64
    # vision hidden size: 1024
    vision_num_key_value_heads = llava_conf.vision_config.hidden_size // llava_conf.vision_config.num_attention_heads
    # we need to subtract 1 from the number of hidden layers because the last layer is not used in LLaVA
    # note; llava_conf.vision_feature_layer is -2 in default config
    vision_layer_num = llava_conf.vision_config.num_hidden_layers + (llava_conf.vision_feature_layer + 1)
    for i_layer in range(vision_layer_num):
        message = queue_get(f"vision transformer layer {i_layer}")
        for key, value in message.items():
            if key == "name":
                continue
        suffix = f'model.vision_tower.vision_model.encoder.layers.{i_layer}.'
        set_hf_param(suffix + 'layer_norm1', message["input norm weight"])
        set_hf_param(suffix + 'layer_norm1', message["input norm bias"], bias=True)
        set_hf_param(suffix + 'layer_norm2', message["pre norm weight"])
        set_hf_param(suffix + 'layer_norm2', message["pre norm bias"], bias=True)

        set_hf_param(suffix + 'mlp.fc1', message["mlp l0 weight"])
        set_hf_param(suffix + 'mlp.fc2', message["mlp l1 weight"])
        qkv_weight = message["qkv weight"]

        try:
            nh = 16
            dim = 64
            qkv_concat = qkv_weight.view(nh, 3 * dim, -1)
            q_reshaped = qkv_concat[:, :dim, :].contiguous().view(-1, qkv_weight.size(-1))
            k_reshaped = qkv_concat[:, dim:2 * dim, :].contiguous().view(-1, qkv_weight.size(-1))
            v_reshaped = qkv_concat[:, 2 * dim:, :].contiguous().view(-1, qkv_weight.size(-1))
            set_hf_param(suffix + 'self_attn.q_proj', q_reshaped)
            set_hf_param(suffix + 'self_attn.k_proj', k_reshaped)
            set_hf_param(suffix + 'self_attn.v_proj', v_reshaped)
            set_hf_param(suffix + 'self_attn.out_proj', message["dense weight"])
        except Exception as e:
            print(f"saver_llava3_hf | Since we are not going to use other vision model, we hardcoded the shape of qkv weight")
            print(f"saver_llava3_hf | If you encounter this error, please check the shape of qkv weight you're trying to load")
            raise e

        # set bias
        qkv_bias = message["qkv bias"]

        qkv_bias = qkv_bias.view(16, -1)  # (16, 192)
        q_bias, k_bias, v_bias = torch.split(qkv_bias, qkv_bias.size(-1) // 3, dim=1)  # (16, 64) x 3
        q_bias = q_bias.reshape(-1)
        k_bias = k_bias.reshape(-1)
        v_bias = v_bias.reshape(-1)

        set_hf_param(suffix + 'self_attn.q_proj', q_bias, bias=True)
        set_hf_param(suffix + 'self_attn.k_proj', k_bias, bias=True)
        set_hf_param(suffix + 'self_attn.v_proj', v_bias, bias=True)
        set_hf_param(suffix + 'self_attn.out_proj', message["dense bias"], bias=True)
        set_hf_param(suffix + 'mlp.fc1', message["mlp l0 bias"], bias=True)
        set_hf_param(suffix + 'mlp.fc2', message["mlp l1 bias"], bias=True)

    # 2. Adapter
    message = queue_get("adapter")
    set_hf_param('model.multi_modal_projector.linear_1', message["linear fc1 weight"])
    set_hf_param('model.multi_modal_projector.linear_2', message["linear fc2 weight"])
    set_hf_param('model.multi_modal_projector.linear_1', message["linear fc1 bias"], bias=True)
    set_hf_param('model.multi_modal_projector.linear_2', message["linear fc2 bias"], bias=True)

    # 3. Decoder
    # 3.1 Text Embeddings
    message = queue_get("language embeddings")
    set_hf_param('model.language_model.model.embed_tokens', message["word embeddings"])

    # 3.2 Decoder Transformer Blocks
    for i_layer in range(text_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.language_model.model.layers.{i_layer}.'
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["pre mlp norm weight"])
        set_hf_param(suffix + 'mlp.gate_proj', message["mlp l0 weight W"])
        set_hf_param(suffix + 'mlp.up_proj', message["mlp l0 weight V"])
        qkv_weight = message["qkv weight"]

        qkv_weight = qkv_weight.view(text_conf.num_key_value_heads, -1, text_conf.hidden_size)
        qkv_weight = torch.split(qkv_weight, [
            text_conf.hidden_size // text_conf.num_key_value_heads,
            text_conf.hidden_size // text_conf.num_attention_heads,
            text_conf.hidden_size // text_conf.num_attention_heads
        ], dim=1)

        q_proj_weight = qkv_weight[0].reshape(-1, text_conf.hidden_size)
        k_proj_weight = qkv_weight[1].reshape(-1, text_conf.hidden_size)
        v_proj_weight = qkv_weight[2].reshape(-1, text_conf.hidden_size)
        set_hf_param(suffix + 'self_attn.q_proj', q_proj_weight)
        set_hf_param(suffix + 'self_attn.k_proj', k_proj_weight)
        set_hf_param(suffix + 'self_attn.v_proj', v_proj_weight)
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
    set_hf_param('model.language_model.model.norm', queue_get('final norm')['weight'])
    set_hf_param('model.language_model.lm_head', queue_get('output layer')['weight'])

    with suspend_nn_inits():
        print("\n\nsaver_llava3_hf | Saving model to disk ...")
        model = LlavaForConditionalGeneration.from_pretrained(
            None,  # type: ignore
            config=llava_conf,
            state_dict=state_dict,
            torch_dtype=torch_dtype,
            output_loading_info=False
        )
        # compare state_dict
        model_state_dict = model.state_dict()
        print("\n\nsaver_llava3_hf | Comparing state_dict:")
        converted_state_dict_keys = set([
            k.replace('model.', '', 1) if k.startswith('model.') else k for k in state_dict.keys()
        ])
        original_state_dict_keys = set(model_state_dict.keys())
        unexpected_keys = converted_state_dict_keys - original_state_dict_keys
        missing_keys = original_state_dict_keys - converted_state_dict_keys
        print(f"\nsaver_llava3_hf | Unexpected keys: {sorted(list(unexpected_keys))}")
        print(f"\nsaver_llava3_hf | Missing keys: {sorted(list(missing_keys))}")

        model.save_pretrained(
            args.save_dir,
            safe_serialization=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer_path
    )
    tokenizer.save_pretrained(args.save_dir)
    print(f"saver_llava3_hf | model saved to {args.save_dir}")
