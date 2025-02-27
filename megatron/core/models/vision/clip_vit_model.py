# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.custom_layers._transformer_engine import TENorm
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


# Note: This is under development and is missing features like position embedding interpolation.
class CLIPViTModel(VisionModule):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        add_class_token: bool = True,
        class_token_len: int = 1,
        output_layer_index: int = -2
    ) -> None:
        super().__init__(config=transformer_config)

        self.output_layer_index = output_layer_index

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        if not self.add_class_token:
            class_token_len = 0
        self.class_token_len = class_token_len

        # self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)
        self.seq_length = self.num_patches + 1  # always add class token since pre-trained model has it

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=False,
        )

        self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
        if not self.add_class_token:
            self.position_ids = self.position_ids[:, :-1]
        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.visual_hidden_size))

        self.ln_pre = TENorm(
            config=self.config,
            hidden_size=self.visual_hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer + final layer norm (via post_process)
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting pipeline parallelism.
        self.transformer = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            disable_pipeline_parallelism=True,
            post_layer_norm=False,  # since we use pre-last layer output
            module_name="clip_vit"
        )

        # Note: a final linear layer present in some implementations is omitted here. It can be added separately where needed.
        # if self.output_layer_index != -1:
        #     # remove final `output_layer_index` layers from the transformer
        #     self.transformer.layers = self.transformer.layers[:self.output_layer_index+1]

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.transformer.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use. If none, all ones.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        x = self.conv1(x)  # shape = [batch, hidden_size, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            x = torch.cat(
                [class_token, x], dim=1
            )  # [batch, grid ** 2 + class_token_len, hidden_size]

        x = x + self.position_embeddings(self.position_ids)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        if attention_mask is None:
            attention_mask = torch.ones(1, 1, x.shape[0], x.shape[0]).cuda()  # [1, 1, s, s]
            attention_mask = attention_mask < 0.5  # to bool
        x = self.transformer(x.contiguous(), attention_mask)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        return x
