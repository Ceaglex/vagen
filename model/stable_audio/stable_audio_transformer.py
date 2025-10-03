# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from math import pi
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor, StableAudioAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableAudioGaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    # Copied from diffusers.models.embeddings.GaussianFourierProjection.__init__
    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            del self.weight
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
            self.weight = self.W
            del self.W

    def forward(self, x):
        if self.log:
            x = torch.log(x)

        x_proj = 2 * np.pi * x[:, None] @ self.weight[None, :]

        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


@maybe_allow_in_graph
class StableAudioDiTBlock(nn.Module):
    r"""
    Transformer block used in Stable Audio model (https://github.com/Stability-AI/stable-audio-tools). Allow skip
    connection and QKNorm

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`): The number of heads to use for the key and value states.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        upcast_attention: bool = False,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, True)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_key_value_attention_heads,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, True)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn="swiglu",
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states


class StableAudioDiTModel(ModelMixin, ConfigMixin):
    """
    The Diffusion Transformer model introduced in Stable Audio.

    Reference: https://github.com/Stability-AI/stable-audio-tools

    Parameters:
        sample_size ( `int`, *optional*, defaults to 1024): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 64): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 24): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`, *optional*, defaults to 12):
            The number of heads to use for the key and value states.
        out_channels (`int`, defaults to 64): Number of output channels.
        cross_attention_dim ( `int`, *optional*, defaults to 768): Dimension of the cross-attention projection.
        time_proj_dim ( `int`, *optional*, defaults to 256): Dimension of the timestep inner projection.
        global_states_input_dim ( `int`, *optional*, defaults to 1536):
            Input dimension of the global hidden states projection.
        cross_attention_input_dim ( `int`, *optional*, defaults to 768):
            Input dimension of the cross-attention projection
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["preprocess_conv", "postprocess_conv", "^proj_in$", "^proj_out$", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.time_proj = StableAudioGaussianFourierProjection(
            embedding_size=time_proj_dim // 2,
            flip_sin_to_cos=True,
            log=False,
            set_W_to_weight=False,
        )

        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        # self.global_proj = nn.Sequential(
        #     nn.Linear(global_states_input_dim, self.inner_dim, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(self.inner_dim, self.inner_dim, bias=False),
        # )

        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),
        )

        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_attention_heads=num_key_value_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        self.gradient_checkpointing = False


    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel.set_default_attn_processor with Hunyuan->StableAudio
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(StableAudioAttnProcessor2_0())


    def pre_forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))
        global_hidden_states = time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)

        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)
        
        return  hidden_states, time_hidden_states, attention_mask, cross_attention_hidden_states, encoder_attention_mask, rotary_embedding


    def block_forward(
        self,
        block_idx: int, 
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cross_attention_hidden_states: torch.FloatTensor = None, 
        encoder_attention_mask: Optional[torch.LongTensor] = None, 
        rotary_embedding: torch.FloatTensor = None
    ):
        block = self.transformer_blocks[block_idx]
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                attention_mask,
                cross_attention_hidden_states,
                encoder_attention_mask,
                rotary_embedding,
            ) # type: ignore
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=cross_attention_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_embedding=rotary_embedding,
            )
        return hidden_states


    def post_forward(
        self,
        hidden_states: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)




    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        # global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`StableAudioDiTModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention
                masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        # cross_attention_hidden_states = encoder_hidden_states
        # global_hidden_states = self.global_proj(global_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        # global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)
        global_hidden_states = time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)

        # prepend global states to hidden states
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cross_attention_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)







class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered


@dataclass
class StableAudioProjectionModelOutput(BaseOutput):
    """
    Args:
    Class for StableAudio projection layer's outputs.
        text_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the hidden-states for the text encoder.
        seconds_start_hidden_states (`torch.Tensor` of shape `(batch_size, 1, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the audio start hidden states.
        seconds_end_hidden_states (`torch.Tensor` of shape `(batch_size, 1, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the audio end hidden states.
    """

    text_hidden_states: Optional[torch.Tensor] = None
    seconds_start_hidden_states: Optional[torch.Tensor] = None
    seconds_end_hidden_states: Optional[torch.Tensor] = None


class StableAudioNumberConditioner(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value

    def forward(
        self,
        floats: torch.Tensor,
    ):
        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (self.max_value - self.min_value)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds


class StableAudioProjectionModel(ModelMixin, ConfigMixin):
    """
    A simple linear projection model to map the conditioning values to a shared latent space.

    Args:
        text_encoder_dim (`int`):
            Dimensionality of the text embeddings from the text encoder (T5).
        conditioning_dim (`int`):
            Dimensionality of the output conditioning tensors.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
    """

    @register_to_config
    def __init__(self, text_encoder_dim, conditioning_dim, min_value, max_value):
        super().__init__()
        self.text_projection = (
            nn.Identity() if conditioning_dim == text_encoder_dim else nn.Linear(text_encoder_dim, conditioning_dim)
        )
        self.start_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)
        self.end_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)

    def forward(
        self,
        text_hidden_states: Optional[torch.Tensor] = None,
        start_seconds: Optional[torch.Tensor] = None,
        end_seconds: Optional[torch.Tensor] = None,
    ):
        text_hidden_states = (
            text_hidden_states if text_hidden_states is None else self.text_projection(text_hidden_states)
        )
        seconds_start_hidden_states = (
            start_seconds if start_seconds is None else self.start_number_conditioner(start_seconds)
        )
        seconds_end_hidden_states = end_seconds if end_seconds is None else self.end_number_conditioner(end_seconds)

        return StableAudioProjectionModelOutput(
            text_hidden_states=text_hidden_states,
            seconds_start_hidden_states=seconds_start_hidden_states,
            seconds_end_hidden_states=seconds_end_hidden_states,
        )




