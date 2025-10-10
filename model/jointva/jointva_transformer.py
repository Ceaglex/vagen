

from typing import Optional, Union, List
import torch
import torch.nn as nn
from safetensors import safe_open
import numpy as np

from diffusers.models.auto_model import AutoModel
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from model.stable_audio.stable_audio_transformer import StableAudioDiTModel, StableAudioProjectionModel
from model.wan.wan_transformer_for_video import WanTransformer3DModel
from diffusers.models.modeling_utils import ModelMixin
from model.jointva.dit_dual_stream import DualDiTBlockAdaLN, CrossDiTBlockAdaLN




def make_video_times(frames: int, tokens_per_frame: int, duration_s: float) -> np.ndarray:  # -> torch.Tensor:
    """
    返回 [L_v] 的秒级时间戳（中心对齐），L_v=frames*tokens_per_frame
    """
    # frame_idx = torch.arange(frames, dtype=dtype)
    # frame_centers = (frame_idx + 0.5) * (duration_s / frames)
    # times = frame_centers.repeat_interleave(tokens_per_frame)

    frame_idx = np.arange(frames)
    frame_centers = (frame_idx + 0.5) * (duration_s / frames)
    times = np.repeat(frame_centers, tokens_per_frame)
    return times


def make_audio_times(length: int, rate_hz: float) -> np.ndarray:   # -> torch.Tensor:
    """
    返回 [L_a] 的秒级时间戳（中心对齐），L_a=length，rate_hz 为音频token率(Hz)
    """
    # idx = torch.arange(length, dtype=dtype)
    # times = (idx + 0.5) / rate_hz

    idx = np.arange(length)
    times = (idx + 0.5) / rate_hz
    return times



class JointVADiTModel(ModelMixin):
    _supports_gradient_checkpointing = True


    
    def __init__(self, 
            load_dtype = torch.float32,
            audio_transformer_path: Optional[str] = None, 
            audio_transformer_weights_path: Optional[str] = None,
            video_transformer_path: Optional[str] = None,
            video_transformer_weights_path: Optional[str] = None,
            bridge_config: Optional[dict] = None,
            bridge_weights_path:  Optional[str] = None,
        ):
        super().__init__()
        self.load_dtype = load_dtype

        # Audio Model
        self.audio_transformer = self._init_class(StableAudioDiTModel, audio_transformer_path, subfolder="transformer")
        if audio_transformer_weights_path is not None:
            self._load_weights(self.audio_transformer, audio_transformer_weights_path)
        # Video Model 
        self.video_transformer = self._init_class(WanTransformer3DModel, video_transformer_path, subfolder="transformer")  
        if video_transformer_weights_path is not None:
            self._load_weights(self.video_transformer, video_transformer_weights_path)


        # Bridge Model
        if bridge_config is not None:
            self.dual_block_nums = bridge_config["dual_dit_configs"].get('num_dual_dit_blocks', 3)
            self.num_channels_1 = bridge_config["dual_dit_configs"].get('num_input_channels_1', 1536)
            self.num_channels_2 = bridge_config["dual_dit_configs"].get('num_input_channels_2', 1536)
            self.num_qk_channels = bridge_config["dual_dit_configs"].get('num_qk_channels', 1536)
            self.num_v_channels = bridge_config["dual_dit_configs"].get('num_v_channels', 1536)
            self.num_heads = bridge_config["dual_dit_configs"].get('num_heads', 12)
            self.t_emb_dim_1 = bridge_config["dual_dit_configs"].get('t_emb_dim_1', 9216)
            self.t_emb_dim_2 = bridge_config["dual_dit_configs"].get('t_emb_dim_2', 1536)
            self.video_bridge_points = bridge_config['dual_dit_configs'].get('video_bridge_points', [7, 11, 15])
            self.audio_bridge_points = bridge_config['dual_dit_configs'].get('audio_bridge_points', [5, 8, 11])
            self.fusion_type = bridge_config['dual_dit_configs'].get('fusion_type', 'bicross') #self.train_config['fusion_type']
            self.dual_dit_blocks = nn.ModuleList([])


            if self.fusion_type == "full_attn":
                for _ in range(self.dual_block_nums):
                    self.dual_dit_blocks.append(
                        DualDiTBlockAdaLN(
                            self.num_channels_1, 
                            self.num_channels_2,
                            self.num_qk_channels,
                            self.num_v_channels,
                            self.num_heads,
                            self.t_emb_dim_1,
                            self.t_emb_dim_2,
                        )
                    )
            elif self.fusion_type == "bicross" or self.fusion_type == "v2a" or self.fusion_type == "a2v":
                for _ in range(self.dual_block_nums):
                    self.dual_dit_blocks.append(
                        CrossDiTBlockAdaLN(
                            self.num_channels_1, 
                            self.num_channels_2,
                            self.num_qk_channels,
                            self.num_v_channels,
                            self.num_heads,
                            self.t_emb_dim_1,
                            self.t_emb_dim_2,
                            self.fusion_type,
                        )
                    )
            else:
                raise NotImplementedError()


            # #### Regular RoPE ####
            # self.rope_pos_embeds_1d_video = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 32760, use_real=True,)
            # self.rope_pos_embeds_1d_audio = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 117, use_real=True,)
            # #### Regular RoPE ####

            #### Aligned RoPE ####
            # v_pos = np.linspace(0, 110-1, 21, dtype=int)    # 21 frames
            # v_pos = np.repeat(v_pos, 30*52)                 # w*h
            # a_pos = np.linspace(0, 110-1, 110,   dtype=int) # 110
            v_pos = make_video_times(frames = 21, tokens_per_frame = 30 * 52, duration_s = 5.0)
            a_pos = make_audio_times(length = 110, rate_hz = 21.5)
            self.rope_pos_embeds_1d_video = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, v_pos, use_real=True,)
            self.rope_pos_embeds_1d_audio = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, a_pos, use_real=True,)
            #### Aligned RoPE ####
            


            self.v_segments, self.a_segments = [], []
            last_split_point = -1
            for point in self.video_bridge_points:
                segment = (last_split_point + 1, point + 1)
                self.v_segments.append(segment)
                last_split_point = point
            self.v_segments.append((last_split_point + 1, len(self.video_transformer.blocks)))

            last_split_point = -1
            for point in self.audio_bridge_points:
                segment = (last_split_point + 1, point + 1)
                self.a_segments.append(segment)
                last_split_point = point
            self.a_segments.append((last_split_point + 1, len(self.audio_transformer.transformer_blocks)))
            print(f"Video segments: {self.v_segments}")
            print(f"Audio segments: {self.a_segments}")
            # TODO: Weight Init
            self._init_bridge()
            if bridge_weights_path is not None:
                self._load_weights(self.dual_dit_blocks, bridge_weights_path)

    
    def audio_forward(self, 
                    latent_model_input,
                    timestep,
                    encoder_hidden_states, 
                    rotary_embedding,
                    return_dict=False):
        assert self.audio_transformer is not None

        hidden_states, time_hidden_states, attention_mask, cross_attention_hidden_states, encoder_attention_mask, rotary_embedding = self.audio_transformer.pre_forward(
            hidden_states = latent_model_input,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            rotary_embedding = rotary_embedding,
        )

        for block_idx in range(len(self.audio_transformer.transformer_blocks)):
            hidden_states = self.audio_transformer.block_forward(
                block_idx = block_idx,
                hidden_states = hidden_states, 
                attention_mask = attention_mask, 
                cross_attention_hidden_states = cross_attention_hidden_states, 
                encoder_attention_mask = encoder_attention_mask, 
                rotary_embedding = rotary_embedding
            )

        out = self.audio_transformer.post_forward(
            hidden_states = hidden_states,
            return_dict = return_dict
        )
        return out


    def video_forward(self, 
            latent_model_input,
            timestep,
            encoder_hidden_states,
            attention_kwargs=None,
            return_dict=False,
        ):
        assert self.video_transformer is not None

        hidden_states, timestep_proj, encoder_hidden_states, rotary_emb, temb, shape_config, lora_scale = self.video_transformer.pre_forward(
            hidden_states = latent_model_input,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            attention_kwargs = attention_kwargs,
        )

        for block_idx in range(len(self.video_transformer.blocks)):
            hidden_states = self.video_transformer.block_forward(
                block_idx = block_idx,
                hidden_states = hidden_states, 
                encoder_hidden_states = encoder_hidden_states,
                timestep_proj = timestep_proj,
                rotary_emb = rotary_emb,
            )

        out = self.video_transformer.post_forward(
            hidden_states = hidden_states,
            temb = temb,
            shape_config = shape_config,
            lora_scale = lora_scale,
            return_dict = return_dict
        )
        return out


    # joint va generation
    def forward(self, 
                timestep,
                v_latent_model_input,
                v_encoder_hidden_states,
                v_attention_kwargs,
                a_latent_model_input,
                a_encoder_hidden_states, 
                a_rotary_embedding,
                return_dict=False):

        v_hidden_states, v_timestep_proj, v_encoder_hidden_states, v_rotary_emb, v_temb, v_shape_config, v_lora_scale = self.video_transformer.pre_forward(
            hidden_states = v_latent_model_input,
            timestep = timestep,
            encoder_hidden_states = v_encoder_hidden_states,
            attention_kwargs = v_attention_kwargs,
        )
        a_hidden_states, a_time_hidden_states, a_attention_mask, a_cross_attention_hidden_states, a_encoder_attention_mask, a_rotary_embedding = self.audio_transformer.pre_forward(
                    hidden_states = a_latent_model_input,
                    timestep = timestep,
                    encoder_hidden_states = a_encoder_hidden_states,
                    rotary_embedding = a_rotary_embedding,
                )
        

        v_time_emb = v_temb
        a_time_emb = a_time_hidden_states
        for bridge_idx in range(len(self.dual_dit_blocks) + 1):
            v_start, v_end = self.v_segments[bridge_idx]
            a_start, a_end = self.a_segments[bridge_idx]
            for block_idx in range(v_start, v_end):
                v_hidden_states = self.video_transformer.block_forward(
                    block_idx = block_idx,
                    hidden_states = v_hidden_states, 
                    encoder_hidden_states = v_encoder_hidden_states,
                    timestep_proj = v_timestep_proj,
                    rotary_emb = v_rotary_emb,
                )
            for block_idx in range(a_start, a_end):
                a_hidden_states = self.audio_transformer.block_forward(
                    block_idx = block_idx,
                    hidden_states = a_hidden_states, 
                    attention_mask = a_attention_mask, 
                    cross_attention_hidden_states = a_cross_attention_hidden_states, 
                    encoder_attention_mask = a_encoder_attention_mask, 
                    rotary_embedding = a_rotary_embedding
                )
            # # TODO: Uncomment BridgeDiT, Clarify RoPE
            if bridge_idx != len(self.dual_dit_blocks):
                bridge_block = self.dual_dit_blocks[bridge_idx]
                v_hidden_states, a_hidden_states = bridge_block(v_hidden_states, a_hidden_states,  # [bs, 32760, 1536]   [1, 122(22.5 * 5.4), 1536]
                                                                v_time_emb, a_time_emb,            # [1, 1536]  [1, 1536]
                                                                self.rope_pos_embeds_1d_video,     # ([32760, 128], [32760, 128])
                                                                self.rope_pos_embeds_1d_audio)     # ([117, 128], [117, 128])


        v_out = self.video_transformer.post_forward(
            hidden_states = v_hidden_states,
            temb = v_time_emb,
            shape_config = v_shape_config,
            lora_scale = v_lora_scale,
            return_dict = return_dict
        )
        a_out = self.audio_transformer.post_forward(
            hidden_states = a_hidden_states,
            return_dict = return_dict
        )
        return v_out, a_out


    def _init_class(self, 
                    class_name, 
                    path = None,
                    subfolder = None,
                    device = None):
        if path is None:
            return None
        
        kwargs = {
            'torch_dtype': self.load_dtype,
            'local_files_only': True, 
            'use_safetensors': True
        }
        if subfolder is not None:
            kwargs['subfolder'] = subfolder
            
        module = class_name.from_pretrained(path, **kwargs)
        if device is not None:
            module.eval().requires_grad_(False).to(device)
        return module
        

    def _load_weights(self, module, safetensor_path):
        if safetensor_path is not None:
            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as f: 
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            module.load_state_dict(state_dict, strict=True)
    
    def _init_bridge(self):
        for attn_block in self.dual_dit_blocks:
            for name, param in attn_block.named_parameters():
                if 'adaln_modulation' in name:
                    torch.nn.init.constant_(param, 0)
                elif ('weight' or 'bias' in name) and param.dim() >= 2:  # 针对权重矩阵
                    nn.init.xavier_uniform_(param)



