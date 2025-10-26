# import pdb; pdb.set_trace()
import logging, math, os, shutil
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import copy
import soundfile as sf
import json
import time

import random
import torch
import torch.nn.functional as F
import torchaudio
import torch.nn.functional as nn_func
from torch.cuda.amp import autocast
import gc
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger

from transformers import SpeechT5HifiGan
from diffusers.models.auto_model import AutoModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.import_utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, PeftModel, get_peft_model

from worker.base import prepare_config, prepare_everything
from dataset.hy_video_audio import build_video_loader
from utils.va_processing import add_audio_to_video, save_video
from utils.text_encoding import get_t5_prompt_embeds, encode_prompt, encode_prompt_sd, encode_duration_sd, prepare_extra_step_kwargs
from utils.optimizer import get_optimizer, init_weights, init_class, load_weights
from utils.model_loading import (
    init_fusion_score_model_ovi, 
    init_text_model, 
    init_mmaudio_vae, 
    init_wan_vae_2_2, 
    load_fusion_checkpoint
)
from utils.fm_solvers import FlowUniPCMultistepScheduler
from utils.va_processing import snap_hw_to_multiple_of_32

logger = get_logger(__name__)




def log_validation(
    config,
    video_config,
    audio_config,
    fusion_model,
    vae_model_video,
    vae_model_audio,
    text_model,
    infer_dtype,
    accelerator,
    global_step
    ):

    with open(config.prompt_index_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = list(data.items())
    idx = accelerator.process_index if accelerator.process_index < len(data) else 0
    step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
    sync_times = len(data) // step_range
    data = data[idx::step_range]
    output_dir = os.path.join(accelerator.project_dir, args.logging_subdir, str(global_step))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving path :{output_dir}")
    
    # Initialize generators
    v_generator = torch.Generator(accelerator.device).manual_seed(config.seed) if config.seed is not None else None
    a_generator = torch.Generator(accelerator.device).manual_seed(config.seed) if config.seed is not None else None
    
    # Initialize schedulers
    scheduler_video = FlowUniPCMultistepScheduler(num_train_timesteps=config.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    scheduler_audio = FlowUniPCMultistepScheduler(num_train_timesteps=config.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False)

    num_va_per_prompt = config.num_va_per_prompt
    batch_size = config.batch_size
    sample_steps = config.num_inference_steps
    shift = config.scheduler.flow_shift
    v_negative_prompt = config.v_negative_prompt
    a_negative_prompt = config.a_negative_prompt
    # # v_num_frames = config.v_num_frames
    v_height = config.video_size[1]
    v_width = config.video_size[0]
    v_guidance_scale = config.v_guidance_scale
    a_guidance_scale = config.a_guidance_scale
    slg_layer = config.slg_layer
    do_classifier_free_guidance = (a_guidance_scale > 1.0) or (v_guidance_scale > 1.0)
    fusion_model.eval()


    with torch.no_grad():
        for start_idx in range(0, len(data), batch_size):
            end_idx = min(len(data), start_idx + batch_size)
            paths = [data[i][0] for i in range(start_idx, end_idx)]
            infos = [data[i][1] for i in range(start_idx, end_idx)]

            text_prompt = [f"{info['video_caption']} <AUDCAP>{info['audio_caption']}<ENDAUDCAP>" for info in infos] # Use video caption as the main prompt
            print(paths, "----", text_prompt)
            # text_prompt = f"{info['video_caption']} <AUDCAP>{info['audio_caption']}<ENDAUDCAP>" # Use video caption as the main prompt
        
            video_h, video_w = v_height, v_width
            snap_area = max(video_h * video_w, 720 * 720)
            video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area=snap_area)
            video_latent_h, video_latent_w = video_h // 16, video_w // 16
                    
            audio_latent_channel = audio_config['in_dim'] 
            video_latent_channel = video_config['in_dim'] 
            audio_latent_length = 157  # Fixed for Ovi model
            video_latent_length = 31   # Fixed for Ovi model


            text_embeddings = text_model(text_prompt + [v_negative_prompt, a_negative_prompt], text_model.device)
            text_embeddings = [emb.to(infer_dtype).to(accelerator.device) for emb in text_embeddings]
            text_embeddings_audio_pos = text_embeddings[ : batch_size]
            text_embeddings_video_pos = text_embeddings[ : batch_size]
            text_embeddings_video_neg = text_embeddings[batch_size]
            text_embeddings_audio_neg = text_embeddings[batch_size + 1]

            # Initialize latents
            video_noise = [ torch.randn((video_latent_channel, video_latent_length, video_latent_h, video_latent_w), 
                            device=accelerator.device, dtype=infer_dtype, generator=v_generator)] * batch_size
            video_noise = torch.stack(video_noise, dim = 0)
            audio_noise = [ torch.randn((audio_latent_length, audio_latent_channel), 
                            device=accelerator.device, dtype=infer_dtype, generator=a_generator)] * batch_size
            audio_noise = torch.stack(audio_noise, dim = 0)

            # Calculate sequence lengths
            max_seq_len_audio = audio_noise.shape[-2]
            patch_size = fusion_model.module.video_model.patch_size if hasattr(fusion_model, "module") else fusion_model.video_model.patch_size
            _patch_size_h, _patch_size_w = patch_size[1], patch_size[2]
            max_seq_len_video = video_noise.shape[-1] * video_noise.shape[-2] * video_noise.shape[-3] // (_patch_size_h * _patch_size_w)


            with torch.amp.autocast('cuda', enabled=True, dtype=infer_dtype):
                scheduler_video.set_timesteps(sample_steps, device=accelerator.device, shift=shift)
                timesteps_video = scheduler_video.timesteps
                scheduler_audio.set_timesteps(sample_steps, device=accelerator.device, shift=shift)
                timesteps_audio = scheduler_audio.timesteps

                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio)), 
                                        total=len(timesteps_video), desc="Sampling"):
                    # print(video_noise.mean(), video_noise[0], audio_noise.mean(), audio_noise[0])
                    timestep_input = torch.full((batch_size,), t_v, device=accelerator.device)
                            
                    # Forward pass with positive prompts
                    pos_forward_args = {
                        'audio_context': text_embeddings_audio_pos,
                        'vid_context': text_embeddings_video_pos,
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': False,
                    }
                    pred_video_guided, pred_audio_guided = fusion_model(
                        vid=[i for i in video_noise],
                        audio=[i for i in audio_noise],
                        t=timestep_input,
                        **pos_forward_args
                    )
                    pred_video_guided = [i.to(infer_dtype) for i in pred_video_guided]
                    pred_audio_guided = [i.to(infer_dtype) for i in pred_audio_guided]
                            

                    # Forward pass with negative prompts for classifier-free guidance
                    if do_classifier_free_guidance:
                        neg_forward_args = {
                            'audio_context': [text_embeddings_audio_neg for _ in range(batch_size)],
                            'vid_context': [text_embeddings_video_neg for _ in range(batch_size)],
                            'vid_seq_len': max_seq_len_video,
                            'audio_seq_len': max_seq_len_audio,
                            'first_frame_is_clean': False,
                            'slg_layer': slg_layer
                        }  
                        pred_vid_neg, pred_audio_neg = fusion_model(
                            vid=[i for i in video_noise],
                            audio=[i for i in audio_noise],
                            t=timestep_input,
                            **neg_forward_args
                        )
                        pred_vid_neg   = [i.to(infer_dtype) for i in pred_vid_neg]
                        pred_audio_neg = [i.to(infer_dtype) for i in pred_audio_neg]

                        pred_video_guided = [pred_vid_neg[i]   + v_guidance_scale * (pred_video_guided[i] - pred_vid_neg[i])   for i in range(batch_size)]
                        pred_audio_guided = [pred_audio_neg[i] + a_guidance_scale * (pred_audio_guided[i] - pred_audio_neg[i]) for i in range(batch_size)]

                    # Update latents using schedulers
                    pred_video_guided = torch.stack(pred_video_guided, dim = 0)
                    pred_audio_guided = torch.stack(pred_audio_guided, dim = 0)


                    video_noise = scheduler_video.step(pred_video_guided, t_v, video_noise, return_dict=False)[0]
                    audio_noise = scheduler_audio.step(pred_audio_guided, t_a, audio_noise, return_dict=False)[0]

            # Decode audio
            audio_latents_for_vae = audio_noise.transpose(1, 2)  # 1, c, l
            generated_audio = vae_model_audio.wrapped_decode(audio_latents_for_vae)
            generated_audio = generated_audio.cpu().float().numpy()
                            
            # Decode video
            video_latents_for_vae = video_noise    # 1, c, f, h, w
            generated_video = vae_model_video.wrapped_decode(video_latents_for_vae)
            generated_video = generated_video.cpu().float().numpy()  # c, f, h, w

            # Save results
            for i in range(batch_size):
                name = paths[i].split('/')[-1].replace('.mp4', '')
                v_path = f"{output_dir}/{name}.mp4"                
                save_video(v_path, generated_video[i], generated_audio[i][0], fps=config.data_info.video_info.fps, sample_rate=config.data_info.audio_info.sr)

            if sync_times != 0: 
                sync_times -= 1
                accelerator.wait_for_everyone()
                print(accelerator.device, sync_times)


        # Clear cache after validation
        torch.cuda.empty_cache()
        gc.collect()
        free_memory()
    fusion_model.train()







def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    infer_dtype = torch.bfloat16

    # Initialize Ovi fusion model
    fusion_model, video_config, audio_config = init_fusion_score_model_ovi(rank=accelerator.device, meta_init=True)
    if args.fusion_checkpoint_path is not None:
        load_fusion_checkpoint(fusion_model, checkpoint_path=args.fusion_checkpoint_path, from_meta=True)
    fusion_model = fusion_model.requires_grad_(False).to(dtype=load_dtype).to(device=accelerator.device).eval()
    fusion_model.set_rope_params()
    

    # Initialize VAE models
    vae_model_video = init_wan_vae_2_2(args.ckpt_dir, rank=accelerator.device)
    vae_model_video.model.requires_grad_(False).eval().to(load_dtype)
    vae_model_video.dtype = load_dtype
    vae_model_audio = init_mmaudio_vae(args.ckpt_dir, rank=accelerator.device)
    vae_model_audio.requires_grad_(False).eval().to(load_dtype)
    
    # Initialize text model
    text_model = init_text_model(args.ckpt_dir, rank=accelerator.device, cpu_offload=False)



    """ ****************************  Optimization setting.  **************************** """

    print(f"Loading LoRA weights from {args.lora_config.lora_load_path}")
    # lora_config = LoraConfig(
    #     r=args.lora_config.rank,
    #     lora_alpha=args.lora_config.lora_alpha,
    #     target_modules=list(args.lora_config.lora_modules),
    #     lora_dropout=0.1,
    # )
    # fusion_model = get_peft_model(fusion_model, lora_config, adapter_name="learner")
    if args.lora_config.lora_load_path is not None:
        fusion_model = PeftModel.from_pretrained(
            fusion_model, 
            args.lora_config.lora_load_path,
            adapter_name="learner"
        )
    fusion_model.set_adapter("learner")
    fusion_model.requires_grad_(False)


    log_validation(
        config=args.validation,
        video_config=video_config,
        audio_config=audio_config,
        fusion_model=fusion_model,
        vae_model_video=vae_model_video,
        vae_model_audio=vae_model_audio,
        text_model=text_model,
        infer_dtype=infer_dtype,
        accelerator=accelerator,
        global_step=args.save_name
    )


if __name__ == "__main__":
    args = prepare_config()
    args, accelerator = prepare_everything(args)
    main(args, accelerator)
