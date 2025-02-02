from copy import deepcopy

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec_clipvit,
    get_gpt_layer_with_transformer_engine_spec_llama,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.training import get_args, print_rank_0
from megatron.training.activations import quick_gelu
from megatron.training.arguments import core_transformer_config_from_args
from tools.checkpoint.loader_llava_hf import load_args_from_checkpoint


def get_image_token_count():
    args = get_args()

    add_class_token = args.add_class_token

    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_tokens = num_patches + (1 if add_class_token else 0)
    return num_image_tokens


def model_provider(add_encoder: bool = True, pre_process=True, post_process=True) -> LLaVAModel:
    """Builds the model.

    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.

    Args:
        add_encoder (bool): Whether to include the vision encoder in the model.
        pre_process (bool): Enable preprocessing in the model. NOTE: Not used at the moment.
        post_process (bool): Enable postprocessing in the model. NOTE: Not used at the moment.

    Returns:
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): A multimodal model
    """
    parallel_state.get_pipeline_model_parallel_rank()
    args = get_args()

    args.untie_embeddings_and_output_weights = True
    args.masked_softmax_fusion = False
    args.bias_gelu_fusion = False
    args.bias_dropout_fusion = False
    args.hidden_dropout = 0.0
    args.attention_softmax_in_fp32 = True

    args.vision_activation_func = quick_gelu

    num_image_tokens = get_image_token_count()
    args.decoder_seq_length = args.seq_length + num_image_tokens

    load_args_from_checkpoint(args)

    print_rank_0('building a multimodal model ...')
    language_transformer_config = core_transformer_config_from_args(get_args())

    if args.spec is not None:
        raise ValueError("args.spec must be None")
        # language_transformer_layer_spec = import_module(args.spec)
    else:
        # args.spec is actually None
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_llama(
            args.num_experts, args.moe_grouped_gemm, use_fixed_attn_mask=args.use_fixed_attn_mask
        )

    # TODO: Make these configurable via input .yaml config.
    # vision_transformer_config = deepcopy(language_transformer_config)
    # vision_transformer_layer_spec = deepcopy(language_transformer_layer_spec)
    if args.vision_model_subtype == "clip" or args.vision_model_subtype is None:
        print("Using CLIP vision model")
        vision_transformer_config = core_transformer_config_from_args(args, prefix="vision_")
        vision_transformer_config.num_layers = vision_transformer_config.num_layers - 1  # remove the last layer
        vision_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_clipvit(
            args.num_experts, args.moe_grouped_gemm
        )
    elif args.vision_model_subtype == "siglip":
        print("Using Siglip vision model")
        vision_transformer_config = core_transformer_config_from_args(args, prefix="vision_")
        vision_transformer_config.num_layers = vision_transformer_config.num_layers - 1
        vision_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_clipvit(
            args.num_experts, args.moe_grouped_gemm
        )
    else:
        raise ValueError(f"Invalid vision model subtype: {args.vision_model_subtype}")

    vision_projection_type = "mlp"
    vision_projection_config = core_transformer_config_from_args(args, prefix="projector_")
    vision_projection_modules = deepcopy(language_transformer_layer_spec.submodules.mlp.submodules)

    model = LLaVAModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        # vocab_size=args.padded_vocab_size,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_position_embeddings,
        # max_sequence_length=128,  # for debug, set to small number (e.g., 128)
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_modules,
        vision_projection_type=vision_projection_type,
        add_encoder=add_encoder,
        pre_process=pre_process,
        post_process=post_process,
        language_position_embedding_type='rope',  # this value is not used
        drop_vision_class_token=False,
    )

    # print_rank_0(f"\n\ndebug; model: {model}")

    model.freeze(freeze_language_model=args.freeze_LM, freeze_vision_model=True, freeze_vision_projection=False)

    return model
