import os
import json
from typing import Optional, Union, List
from tqdm.auto import tqdm
import torch
import torchaudio
from torch.cuda.amp import autocast
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from safetensors import safe_open

import gc
from diffusers.video_processor import VideoProcessor
from diffusers.training_utils import free_memory
from diffusers.models.auto_model import AutoModel
from model.wan.pipeline_wan_ttv import AutoencoderKLWan
from utils.text_encoding import get_t5_prompt_embeds, encode_prompt, encode_prompt_sd, encode_duration_sd, prepare_extra_step_kwargs
from model.stable_audio.stable_audio_transformer import StableAudioDiTModel, StableAudioProjectionModel
from model.jointva.jointva_transformer import JointVADiTModel



class JointVAPipeline:
    def __init__(self, 
            load_dtype = torch.float32,
            infer_dtype = torch.float16,
            device = "cpu",
            bridge_config: Optional[dict] = None,
            bridge_weights_path: Optional[str] = None,
            audio_vae_path: Optional[str] = None,
            audio_transformer_path: Optional[str] = None,
            audio_projection_model_path: Optional[str] = None,
            audio_text_encoder_path: Optional[str] = None,
            audio_tokenizer_path: Optional[str] = None,
            audio_transformer_weights_path: Optional[str] = None,
            video_vae_path: Optional[str] = None,
            video_transformer_path: Optional[str] = None,
            video_transformer_weights_path: Optional[str] = None,
            video_text_encoder_path: Optional[str] = None,
            video_tokenizer_path: Optional[str] = None,
            scheduler_config: Optional[dict] = None
        ):

        # Param
        self.load_dtype = load_dtype
        self.infer_dtype = infer_dtype
        self.device = device
        self.v_step_scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction', 
            use_flow_sigmas=True, 
            num_train_timesteps=scheduler_config.num_train_timesteps, 
            flow_shift=scheduler_config.flow_shift
        )
        self.a_step_scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction', 
            use_flow_sigmas=True, 
            num_train_timesteps=scheduler_config.num_train_timesteps, 
            flow_shift=scheduler_config.flow_shift
        )

        # Joint VA DiT
        self.joint_va = JointVADiTModel(
            load_dtype = self.load_dtype,
            audio_transformer_path = audio_transformer_path, 
            audio_transformer_weights_path = audio_transformer_weights_path,
            video_transformer_path = video_transformer_path,
            video_transformer_weights_path = video_transformer_weights_path,
            bridge_config = bridge_config,
            bridge_weights_path = bridge_weights_path
        ).to(self.device)

        # Audio Model Components
        self.audio_vae = self._init_class(AutoModel, audio_vae_path, subfolder="vae", device=self.device)
        self.audio_projection_model = self._init_class(StableAudioProjectionModel, audio_projection_model_path, subfolder="projection_model", device=self.device)
        self.audio_text_encoder = self._init_class(AutoModel, audio_text_encoder_path, subfolder="text_encoder", device=self.device)
        self.audio_tokenizer = self._init_class(AutoModel, audio_tokenizer_path, subfolder="tokenizer")

        # Video Model Components
        self.video_vae = self._init_class(AutoencoderKLWan, video_vae_path, subfolder="vae", device=self.device)
        self.video_processor = VideoProcessor(vae_scale_factor=self.video_vae.config.scale_factor_spatial) if self.video_vae is not None else None
        self.video_text_encoder = self._init_class(AutoModel, video_text_encoder_path, subfolder="text_encoder", device=self.device)
        self.video_tokenizer = self._init_class(AutoModel, video_tokenizer_path, subfolder="tokenizer")


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


    def text_to_audio(self, 
                      a_prompts: list, 
                      a_generator: Optional[torch.Generator] = None,
                      a_negative_prompt: str = "",
                      a_duration: float = 10.0, 
                      a_guidance_scale: float = 7.0,
                      a_num_waveforms_per_prompt: int = 1,
                      a_num_inference_steps: int = 100,
                      **kwargs,
                      ):


        with torch.no_grad():
            with autocast(dtype=self.infer_dtype):
                results = []
                a_do_classifier_free_guidance = a_guidance_scale > 1.0
                num_channels_vae = self.joint_va.audio_transformer.module.config.in_channels if hasattr(self.joint_va.audio_transformer, "module") else self.joint_va.audio_transformer.config.in_channels
                waveform_length = int(a_duration * 22.5)
                shape = (a_num_waveforms_per_prompt, num_channels_vae, waveform_length)

                for prompt in a_prompts:
                    prompt_embeds, negative_prompt_embeds = encode_prompt_sd(
                        prompt,
                        self.audio_tokenizer,
                        self.audio_text_encoder,
                        self.audio_projection_model,
                        self.device,
                        a_do_classifier_free_guidance,
                        a_negative_prompt,
                    ) # type: ignore
                    latents = randn_tensor(shape, generator=a_generator, device=self.device, dtype=self.infer_dtype)

                    rotary_embed_dim = self.joint_va.audio_transformer.module.config.attention_head_dim // 2 if hasattr(self.joint_va.audio_transformer, "module") else self.joint_va.audio_transformer.config.attention_head_dim // 2
                    rotary_embedding = get_1d_rotary_pos_embed(
                        rotary_embed_dim,
                        latents.shape[2] + 1,
                        use_real=True,
                        repeat_interleave_real=False,
                    )

                    self.a_step_scheduler.set_timesteps(a_num_inference_steps, device=self.device)
                    timesteps = self.a_step_scheduler.timesteps
                    for i, t in tqdm(enumerate(timesteps)):

                        a_latent_model_input = latents
                        a_timestep = torch.stack([t for _ in range(a_latent_model_input.shape[0])])
                        a_noise_pred = self.joint_va.audio_forward(
                            latent_model_input = a_latent_model_input,
                            timestep = a_timestep,
                            encoder_hidden_states=prompt_embeds, 
                            rotary_embedding=rotary_embedding,
                            return_dict=False,
                        )[0]
                        if a_do_classifier_free_guidance:
                            a_noise_uncond = self.joint_va.audio_forward(
                                latent_model_input = a_latent_model_input,
                                timestep = a_timestep,
                                encoder_hidden_states=negative_prompt_embeds, 
                                rotary_embedding=rotary_embedding,
                                return_dict=False,
                            )[0]
                            a_noise_pred = a_noise_uncond + a_guidance_scale * (a_noise_pred - a_noise_uncond)
                        latents = self.a_step_scheduler.step(a_noise_pred, t, latents).prev_sample

                    gen_audio = self.audio_vae.decode(latents.to(self.audio_vae.dtype)).sample
                    results.append(gen_audio)
                
            return results

    
    def text_to_video(self, 
                      v_prompts: list, 
                      v_generator: Optional[torch.Generator] = None,
                      v_negative_prompt: str = "",
                      v_num_frames: float = 81, 
                      v_height : int = 480,
                      v_width : int = 832,
                      v_guidance_scale: float = 5.0,
                      v_num_videos_per_prompt: int = 1,
                      v_num_inference_steps: int = 40,
                      **kwargs,
                      ):

        
        with torch.no_grad():
            with autocast(dtype=self.infer_dtype):
                results = []
                do_classifier_free_guidance = v_guidance_scale > 1.0
                num_latent_frames = (v_num_frames - 1) // self.video_vae.config.scale_factor_temporal + 1
                num_channels = self.joint_va.video_transformer.module.config.in_channels if hasattr(self.joint_va.video_transformer, "module") else self.joint_va.video_transformer.config.in_channels
                shape = (
                    v_num_videos_per_prompt,
                    num_channels,
                    num_latent_frames,
                    v_height // self.video_vae.config.scale_factor_spatial,
                    v_width // self.video_vae.config.scale_factor_spatial,
                )

                for prompt in v_prompts:
                    torch.cuda.empty_cache()
                    gc.collect()
                    free_memory()

                    prompt_embeds, negative_prompt_embeds = None, None
                    prompt_embeds, negative_prompt_embeds = encode_prompt(
                        prompt=prompt,
                        negative_prompt=v_negative_prompt,
                        tokenizer = self.video_tokenizer,
                        text_encoder = self.video_text_encoder,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_videos_per_prompt=v_num_videos_per_prompt,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        max_sequence_length=512,
                        device=self.device,
                        dtype=self.infer_dtype, # 必须是float32 / bfloat16，float16有问题，不知道为何
                    ) # type: ignore
                    latents = randn_tensor(shape, generator=v_generator, device=self.device, dtype=self.infer_dtype)
                    
                    
                    self.v_step_scheduler.set_timesteps(v_num_inference_steps, device=self.device)
                    timesteps = self.v_step_scheduler.timesteps
                    for i, t in tqdm(enumerate(timesteps)):

                        latent_model_input = latents
                        timestep = t.expand(latents.shape[0])

                        noise_pred = self.joint_va.video_forward(
                            latent_model_input=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0] 

                        if v_guidance_scale > 1.0:
                            noise_uncond = self.joint_va.video_forward(
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=None,
                                return_dict=False,
                            )[0] 
                            noise_pred = noise_uncond + v_guidance_scale * (noise_pred - noise_uncond)
                        latents = self.v_step_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                            
                    latents = latents.to(self.video_vae.dtype)
                    latents_mean = torch.tensor(self.video_vae.config.latents_mean).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(self.device, self.video_vae.dtype)
                    latents_std = 1 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(self.device, self.video_vae.dtype)
                    latents = latents / latents_std + latents_mean
                    gen_video = self.video_vae.decode(latents, return_dict=False)[0]
                    gen_video = self.video_processor.postprocess_video(gen_video.to(self.load_dtype), output_type='np')
                    results.append(gen_video)
                    # for i in range(num_videos_per_prompt):
                    #     export_to_video(video[i], f"{output_dir}/output{prompt_idx}_{i}.mp4", fps=config.fps if hasattr(config, "fps") else 16)
        
        torch.cuda.empty_cache()
        gc.collect()
        free_memory()
        return results


    def text_to_video_audio(self, 
                            va_prompts: list[str],
                            num_va_per_prompt: int = 1,
                            num_inference_steps: int = 40,

                            v_generator: Optional[torch.Generator] = None,
                            v_negative_prompt: str = "",
                            v_num_frames: float = 81, 
                            v_height : int = 480,
                            v_width : int = 832,
                            v_guidance_scale: float = 5.0,

                            a_generator: Optional[torch.Generator] = None,
                            a_negative_prompt: str = "",
                            a_duration: float = 10.0, 
                            a_guidance_scale: float = 7.0,

                            **kwargs,):
        

        with torch.no_grad():
            with autocast(dtype=self.infer_dtype):
                results = []
                do_classifier_free_guidance = (a_guidance_scale > 1.0) or (v_guidance_scale > 1.0)

                a_num_channels_vae = self.joint_va.audio_transformer.module.config.in_channels if hasattr(self.joint_va.audio_transformer, "module") else self.joint_va.audio_transformer.config.in_channels
                waveform_length = int(a_duration * 21.5)
                a_shape = (num_va_per_prompt, a_num_channels_vae, waveform_length)
                a_rotary_embed_dim = self.joint_va.audio_transformer.module.config.attention_head_dim // 2 if hasattr(self.joint_va.audio_transformer, "module") else self.joint_va.audio_transformer.config.attention_head_dim // 2
                a_rotary_embedding = get_1d_rotary_pos_embed(
                    a_rotary_embed_dim,
                    a_shape[2] + 1,
                    use_real=True,
                    repeat_interleave_real=False,
                )
                
                v_num_channels_vae = self.joint_va.video_transformer.module.config.in_channels if hasattr(self.joint_va.video_transformer, "module") else self.joint_va.video_transformer.config.in_channels
                num_latent_frames = (v_num_frames - 1) // self.video_vae.config.scale_factor_temporal + 1
                v_shape = (
                    num_va_per_prompt,
                    v_num_channels_vae,
                    num_latent_frames,
                    v_height // self.video_vae.config.scale_factor_spatial,
                    v_width // self.video_vae.config.scale_factor_spatial,
                )


                for v_prompt, a_prompt in va_prompts:
                    torch.cuda.empty_cache()
                    gc.collect()
                    free_memory()

                    v_prompt_embeds, v_negative_prompt_embeds = encode_prompt(
                        prompt=v_prompt,
                        negative_prompt=v_negative_prompt,
                        tokenizer = self.video_tokenizer,
                        text_encoder = self.video_text_encoder,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_videos_per_prompt=num_va_per_prompt,
                        prompt_embeds=None,
                        negative_prompt_embeds=None,
                        max_sequence_length=512,
                        device=self.device,
                        dtype=torch.float32, # 必须是float32 / bfloat16，float16有问题，不知道为何
                    ) # type: ignore
                    a_prompt_embeds, a_negative_prompt_embeds = encode_prompt_sd(
                        a_prompt,
                        self.audio_tokenizer,
                        self.audio_text_encoder,
                        self.audio_projection_model,
                        self.device,
                        do_classifier_free_guidance,
                        a_negative_prompt,
                    ) # type: ignore


                    a_latents = randn_tensor(a_shape, generator=v_generator, device=self.device, dtype=self.infer_dtype)
                    v_latents = randn_tensor(v_shape, generator=a_generator, device=self.device, dtype=self.infer_dtype)


                    self.v_step_scheduler.set_timesteps(num_inference_steps, device=self.device)
                    self.a_step_scheduler.set_timesteps(num_inference_steps, device=self.device)
                    timesteps = self.v_step_scheduler.timesteps
                    for i, t in tqdm(enumerate(timesteps)):
                        timestep = t.expand(num_va_per_prompt)
                        v_latent_model_input = v_latents
                        a_latent_model_input = a_latents


                        v_noise_pred, a_noise_pred = self.joint_va(
                            timestep=timestep,
                            v_latent_model_input=v_latent_model_input,
                            v_encoder_hidden_states=v_prompt_embeds,
                            v_attention_kwargs=None,
                            a_latent_model_input=a_latent_model_input,
                            a_encoder_hidden_states=a_prompt_embeds, 
                            a_rotary_embedding=a_rotary_embedding,
                            return_dict=False,
                        )
                        v_noise_pred = v_noise_pred[0]
                        a_noise_pred = a_noise_pred[0]


                        if do_classifier_free_guidance:
                            v_noise_uncond, a_noise_uncond = self.joint_va(
                                timestep=timestep,
                                v_latent_model_input=v_latent_model_input,
                                v_encoder_hidden_states=v_negative_prompt_embeds,
                                v_attention_kwargs=None,
                                a_latent_model_input=a_latent_model_input,
                                a_encoder_hidden_states=a_negative_prompt_embeds, 
                                a_rotary_embedding=a_rotary_embedding,
                                return_dict=False,
                            )
                            v_noise_uncond = v_noise_uncond[0]
                            a_noise_uncond = a_noise_uncond[0]

                            v_noise_pred = v_noise_uncond + v_guidance_scale * (v_noise_pred - v_noise_uncond)
                            a_noise_pred = a_noise_uncond + a_guidance_scale * (a_noise_pred - a_noise_uncond)

                        v_latents = self.v_step_scheduler.step(v_noise_pred, t, v_latents, return_dict=False)[0]
                        a_latents = self.a_step_scheduler.step(a_noise_pred, t, a_latents, return_dict=False)[0] 

                            
                    v_latents = v_latents.to(self.video_vae.dtype)
                    v_latents_mean = torch.tensor(self.video_vae.config.latents_mean).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(self.device, self.video_vae.dtype)
                    v_latents_std = 1 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(self.device, self.video_vae.dtype)
                    v_latents = v_latents / v_latents_std + v_latents_mean
                    gen_video = self.video_vae.decode(v_latents, return_dict=False)[0]
                    gen_video = self.video_processor.postprocess_video(gen_video.to(self.load_dtype), output_type='np')
                    gen_audio = self.audio_vae.decode(a_latents.to(self.audio_vae.dtype)).sample
                    results.append([gen_video, gen_audio])

        torch.cuda.empty_cache()
        gc.collect()
        free_memory()
        return results
