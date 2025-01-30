# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List

import torch

from megatron.core import InferenceParams, parallel_state, tensor_parallel
# from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.siglip_vit_model import SigLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args


# Note: This is under development and may be missing features.
class LLaVAModel(MegatronModule):
    """LLaVA multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the language model.
        language_position_embedding_type (str): Type of the positional embedding to use in the language model.
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): maximum sequence length. This is used for positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        vision_projection_config (TransformerConfig): Config for the projection from vision model outputs to language model inputs.
        vision_projection_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        vision_projection_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be missing when loading a checkpoint. Default False.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_position_embedding_type: str,
        vocab_size: int,
        max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        allow_missing_vision_projection_checkpoint: bool = False,
    ) -> None:
        args = get_args()
        self.args = args
        self.use_fixed_attn_mask = args.use_fixed_attn_mask
        self.sequence_parallel = self.args.sequence_parallel
        print(f"LLaVA init; sequence_parallel: {self.sequence_parallel}")
        super().__init__(config=language_transformer_config)

        logging.getLogger(__name__).warning(
            "LLaVA model is under development and may be missing features."
        )

        self.encoder_hidden_state = None
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.share_embeddings_and_output_weights = False
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=args.max_position_embeddings,  # 4096
            pre_process=pre_process,
            post_process=post_process,  # True
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,  # False
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,  # True
            position_embedding_type="rope",  # "rope"
            rotary_percent=args.rotary_percent,  # 1.0
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

        if self.add_encoder:
            if self.args.vision_model_subtype == "clip" or self.args.vision_model_subtype is None:
                self.vision_model = CLIPViTModel(
                    vision_transformer_config, vision_transformer_layer_spec,
                    add_class_token=self.args.add_class_token
                )
            elif self.args.vision_model_subtype == "siglip":
                self.vision_model = SigLIPViTModel(
                    vision_transformer_config, vision_transformer_layer_spec,
                    add_class_token=self.args.add_class_token
                )
            else:
                raise ValueError(f"Vision model subtype {self.args.vision_model_subtype} is not supported.")

            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_transformer_config.hidden_size,  # input size to the projection.
            )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        print("llava_model; freeze model")
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model:
            if 'vision_model' in self.__dict__["_modules"]:
                modules.append(self.vision_model)
        if freeze_vision_projection:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def build_final_labels(self, labels, input_ids):
        pad_token_id = self.args.pad_token_id
        image_token_index = self.args.image_token_id
        ignore_index = -100  # TODO: temporary fix for ignore index
        if self.args.vision_model_subtype == "clip":
            num_image_patches = 576  # TODO: temporary fix for number of image patches
        else:
            num_image_patches = 729
        if self.args.add_class_token:
            if self.args.vision_model_subtype == "clip":
                num_image_patches += 1
            elif self.args.vision_model_subtype == "siglip":
                raise ValueError("Siglip model does not have class token.")
            else:
                raise ValueError(f"Invalid vision model subtype: {self.args.vision_model_subtype}")

        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_labels = torch.full(
            (batch_size, max_embed_dim), ignore_index, dtype=labels.dtype, device=labels.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = labels.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # cut into the original sequence length
        # and shift the labels to the right by one position
        final_labels = final_labels[:, 1:sequence_length]

        return final_labels

    def _merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids, labels, attention_mask
    ):
        pad_token_id = self.args.pad_token_id
        image_token_index = self.args.image_token_id
        # image_features: [num_image_patches, batch_size, embed_dim] -> [batch_size, num_image_patches, embed_dim]
        image_features = image_features.permute(1, 0, 2)
        # inputs_embeds: [text_seq_len, batch_size, embed_dim] -> [batch_size, text_seq_len, embed_dim]
        inputs_embeds = inputs_embeds.permute(1, 0, 2)

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        final_embedding = final_embedding.permute(1, 0, 2)  # [max_embed_dim, batch_size, embed_dim]

        # cut into the original sequence length
        # and shift the labels to the right by one position
        final_embedding = final_embedding[:sequence_length-1]

        if labels is not None:
            final_labels = self.build_final_labels(labels, input_ids)
        else:
            final_labels = None

        if self.sequence_parallel:
            # pad embedding and labels to divisible by tensor model parallel size
            tp_size = self.args.tensor_model_parallel_size
            remainder = final_embedding.size(0) % tp_size
            if remainder != 0:
                padding = tp_size - remainder
                final_embedding = torch.cat([
                    final_embedding,
                    torch.zeros(padding, batch_size, embed_dim, device=final_embedding.device, dtype=inputs_embeds.dtype)
                ], dim=0)
                if final_labels is not None:
                    final_labels = torch.cat([final_labels, torch.full((batch_size, padding), pad_token_id, dtype=final_labels.dtype, device=final_labels.device)], dim=-1)

        # all tokens should attend to all tokens before the end of the image;
        # after that, tri-diagonal attention mask is used
        # Note: A True value means the corresponding position is masked out
        #       and a False means that position is allowed to participate in attention.

        if self.use_fixed_attn_mask:
            final_attention_mask = torch.zeros(
                batch_size, 1, attention_mask.size(-1)-1, attention_mask.size(-1)-1, device=attention_mask.device
            )
            should_attend_size = torch.where(image_to_overwrite)[1][-1].item()
            tril_size = final_attention_mask.size(-1) - should_attend_size
            final_attention_mask[:, :, should_attend_size:, should_attend_size:] = torch.triu(
                torch.ones(tril_size, tril_size), diagonal=0
            )
            final_attention_mask[:, :, :should_attend_size, should_attend_size:] = 1
            final_attention_mask = final_attention_mask.bool()
        else:
            final_attention_mask = attention_mask
        position_ids = None

        return final_embedding, final_labels, final_attention_mask, position_ids

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [batch, img_h, img_w].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        # If running inference, we can skip image token computation if they were computed already earlier for this sample.
        use_inference_kv_cache = (
                inference_params is not None
                and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            image_embeddings = None
            # combined_embeddings = language_embeddings
        elif self.add_encoder:
            image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]
            # image_embeddings = image_embeddings[:, self.vision_model.class_token_len:, :]
            image_embeddings = image_embeddings.permute(1, 0, 2)  # [img_seq_len, b, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict[
                    "image_tokens_count"
                ] = image_embeddings.shape[0]
        else:
            image_embeddings = self.encoder_hidden_state

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            if use_inference_kv_cache:
                combined_embeddings = language_embeddings
            else:
                if self.sequence_parallel:
                    language_embeddings = tensor_parallel.gather_from_sequence_parallel_region(
                        language_embeddings, tensor_parallel_output_grad=False
                    )
                combined_embeddings, labels, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                    image_embeddings, language_embeddings, input_ids, labels, attention_mask
                )
                if self.sequence_parallel:
                    # scatter back to sequence parallel region
                    combined_embeddings = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                        combined_embeddings
                    )
        else:
            combined_embeddings = None
            labels = self.build_final_labels(labels, input_ids)

        if labels is not None:
            labels[labels == -100] = 0
        output = self.language_model(
            None,  # input_ids
            None,  # position_ids
            attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params
        )

        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        incompatible_keys.missing_keys.remove(param_name)
