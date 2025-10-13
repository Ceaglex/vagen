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
                v_path = f"{output_dir}/{paths[i].split('/')[-1][:-4]}.mp4"                
                save_video(v_path, generated_video[i], generated_audio[i][0], fps=config.data_info.video_info.fps, sample_rate=config.data_info.audio_info.sr)
            
        # Clear cache after validation
        torch.cuda.empty_cache()
        gc.collect()
        free_memory()
    fusion_model.train()





def training_step(batch, 
                  fusion_model,
                  vae_model_video,
                  vae_model_audio,
                  text_model,
                  accelerator, 
                  dpo_beta,
                  load_dtype):

    # TODO: Check DPO Loss and dataset W-L data pair order
    text_prompt = batch['v_prompt']  # Use video prompt as main prompt
    text_prompt = [f"{batch['v_prompt'][i]} <AUDCAP>{batch['a_prompt'][i]}<ENDAUDCAP>" for i in range(len(batch['v_prompt']))] # Use video caption as the main prompt
    waveform = batch['audio_waveform'][:,0,:]
    video_pixel = batch['video_pixel']
    waveform_lose = batch['audio_waveform_lose'][:,0,:]
    video_pixel_lose = batch['video_pixel_lose']
    waveform = torch.concat([waveform, waveform_lose], dim = 0)
    video_pixel = torch.concat([video_pixel, video_pixel_lose], dim = 0)
    group_size = len(text_prompt)
    batch_size = group_size * 2


    with torch.no_grad():
        # Encode text
        text_embeddings = text_model(text_prompt, text_model.device)
        text_embeddings = [emb.to(load_dtype).to(accelerator.device) for emb in text_embeddings]

        # Encode video and audio to latents
        v_latents = vae_model_video.wrapped_encode(video_pixel.to(load_dtype)).to(load_dtype)
        a_latents = vae_model_audio.wrapped_encode(waveform.to(torch.float32)).transpose(1, 2).to(load_dtype)

        # # # Check Latent
        # audio_latents_for_vae = a_latents.transpose(1, 2)  # 1, c, l
        # generated_audio = vae_model_audio.wrapped_decode(audio_latents_for_vae)
        # generated_audio = generated_audio.cpu().float().numpy()         
        # video_latents_for_vae = v_latents    # 1, c, f, h, w
        # generated_video = vae_model_video.wrapped_decode(video_latents_for_vae)
        # generated_video = generated_video.cpu().float().numpy()  # c, f, h, w
        # for i in range(batch_size):
        #     v_path = f"test{i}.mp4"                
        #     save_video(v_path, generated_video[i], generated_audio[i][0], fps=24, sample_rate=16000)
            
        # Add noise
        v_noise = torch.randn_like(v_latents).detach()
        a_noise = torch.randn_like(a_latents).detach()
        
        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=accelerator.device).detach()
        # Apply flow matching noise schedule
        t_v = (timesteps / 1000).to(v_latents.dtype).view(-1, *([1] * (v_latents.ndim - 1)))
        t_a = (timesteps / 1000).to(a_latents.dtype).view(-1, *([1] * (a_latents.ndim - 1)))
        
        v_latent_model_input = (1 - t_v) * v_latents + t_v * v_noise
        a_latent_model_input = (1 - t_a) * a_latents + t_a * a_noise
        
        # Calculate targets (flow matching targets)
        v_target = v_noise - v_latents
        a_target = a_noise - a_latents

        # Calculate sequence lengths
        max_seq_len_audio = a_latent_model_input.shape[-2]
        patch_size = fusion_model.module.video_model.patch_size if hasattr(fusion_model, "module") else fusion_model.video_model.patch_size
        _patch_size_h, _patch_size_w = patch_size[1], patch_size[2]
        max_seq_len_video = v_latent_model_input.shape[-1] * v_latent_model_input.shape[-2] * v_latent_model_input.shape[-3] // (_patch_size_h * _patch_size_w)


        # Forward pass through ref fusion model
        fusion_model.module.set_adapter("ref") if hasattr(fusion_model, "module") else fusion_model.set_adapter("ref")
        pred_vid_ref, pred_audio_ref = fusion_model(
            vid=[v_latent_model_input[i] for i in range(batch_size)],
            audio=[a_latent_model_input[i] for i in range(batch_size)],
            t=timesteps,
            vid_context=text_embeddings,
            audio_context=text_embeddings,
            vid_seq_len=max_seq_len_video,
            audio_seq_len=max_seq_len_audio,
            first_frame_is_clean=False
        )
        pred_vid_ref = torch.stack(pred_vid_ref, dim = 0).detach()
        pred_audio_ref = torch.stack(pred_audio_ref, dim = 0).detach()



    # Forward pass through fusion model
    fusion_model.module.set_adapter("learner") if hasattr(fusion_model, "module") else fusion_model.set_adapter("learner")
    pred_vid, pred_audio = fusion_model(
        vid=[v_latent_model_input[i] for i in range(batch_size)],
        audio=[a_latent_model_input[i] for i in range(batch_size)],
        t=timesteps,
        vid_context=text_embeddings,
        audio_context=text_embeddings,
        vid_seq_len=max_seq_len_video,
        audio_seq_len=max_seq_len_audio,
        first_frame_is_clean=False
    )
    pred_vid = torch.stack(pred_vid, dim = 0)
    pred_audio = torch.stack(pred_audio, dim = 0)

    v_model_losses = (pred_vid.float() - v_target.float()).pow(2).mean(dim=[1, 2, 3, 4])
    v_model_losses_w, v_model_losses_l = v_model_losses.chunk(2)
    v_model_diff = v_model_losses_w - v_model_losses_l

    a_model_losses = (pred_audio.float() - a_target.float()).pow(2).mean(dim=[1, 2])
    a_model_losses_w, a_model_losses_l = a_model_losses.chunk(2)    
    a_model_diff = a_model_losses_w - a_model_losses_l
    

    with torch.no_grad():
        v_ref_losses = (pred_vid_ref.float() - v_target.float()).pow(2).mean(dim=[1, 2, 3, 4])
        v_ref_losses_w, v_ref_losses_l = v_ref_losses.chunk(2)
        v_ref_diff = v_ref_losses_w - v_ref_losses_l

        a_ref_losses = (pred_audio_ref.float() - a_target.float()).pow(2).mean(dim=[1, 2])
        a_ref_losses_w, a_ref_losses_l = a_ref_losses.chunk(2)
        a_ref_diff = a_ref_losses_w - a_ref_losses_l


    scale_term = -0.5 * dpo_beta

    v_inside_term = scale_term * (v_model_diff - v_ref_diff)
    # v_implicit_acc = (v_inside_term > 0).sum().float() / v_inside_term.size(0)
    v_dpo_loss = -F.logsigmoid(v_inside_term).mean()

    a_inside_term = scale_term * (a_model_diff - a_ref_diff)
    # a_implicit_acc = (a_inside_term > 0).sum().float() / a_inside_term.size(0)
    a_dpo_loss = -F.logsigmoid(a_inside_term).mean()




    # # Calculate SFT MSE losses
    # loss_v = nn_func.mse_loss(pred_vid.float(), v_target.float(), reduction="mean")
    # loss_a = nn_func.mse_loss(pred_audio.float(), a_target.float(), reduction="mean")


    return v_dpo_loss, a_dpo_loss



def checkpointing_step(model, accelerator, logger, args, ckpt_idx = 0):
    ckpt_dir = os.path.join(accelerator.project_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(ckpt_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)

        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    # Save whole model lora
    accelerator.save_state()
    # Save lora
    lora_save_path = f"{ckpt_dir}/checkpoint_{ckpt_idx}"
    model.module.save_pretrained(lora_save_path) if hasattr(model, "module") else model.save_pretrained(lora_save_path)
    logger.info(f"Saved state to {ckpt_dir}")


def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    infer_dtype = torch.bfloat16

    # Initialize Ovi fusion model
    fusion_model, video_config, audio_config = init_fusion_score_model_ovi(rank=accelerator.device, meta_init=True)
    # fusion_model.video_config = video_config
    # fusion_model.audio_config = audio_config
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

    if args.gradient_checkpointing:
        fusion_model.train()
        fusion_model.gradient_checkpointing = True

    """ ****************************  Weights dtype setting.  **************************** """
    weight_dtype = load_dtype
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")
    
    # if args.mixed_precision == "fp16":
    #     cast_training_params([fusion_model], dtype=torch.float32)

    """ ****************************  Data setting.  **************************** """
    logger.info("=> Preparing training data...", main_process_only=True)
    dataset_cfg = args.hy_dataloader
    train_dataloader = build_video_loader(args=dataset_cfg)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size_local

    """ ****************************  Optimization setting.  **************************** """
    fusion_model.requires_grad_(False)
    if args.lora_config.use_lora == True:
        lora_config = LoraConfig(
            r=args.lora_config.rank,
            lora_alpha=args.lora_config.lora_alpha,
            target_modules=list(args.lora_config.lora_modules),
            lora_dropout=0.1,
            bias="none",)

        # TODO: Check load lora
        if hasattr(args.lora_config, 'lora_load_path') and args.lora_config.lora_load_path:
            print(f"Loading LoRA weights from {args.lora_config.lora_load_path}")
            fusion_model = PeftModel.from_pretrained(
                fusion_model, 
                args.lora_config.lora_load_path,
                adapter_name="learner"
            )
            fusion_model = get_peft_model(fusion_model, lora_config, adapter_name="ref")
        else:
            torch.manual_seed(args.seed); random.seed(args.seed)
            fusion_model = get_peft_model(fusion_model, lora_config, adapter_name="ref")
            torch.manual_seed(args.seed); random.seed(args.seed)
            fusion_model = get_peft_model(fusion_model, lora_config, adapter_name="learner")

        fusion_model.set_adapter("learner")
        fusion_model.print_trainable_parameters()


    if args.block_config.train_va == True:
        for block_idx in args.block_config.audio_trainable_blocks:
            fusion_model.audio_model.blocks[block_idx].requires_grad_(True)
        for block_idx in args.block_config.video_trainable_blocks:
            fusion_model.video_model.blocks[block_idx].requires_grad_(True)
    
    if args.optimize_params is not None:
        for name, param in fusion_model.named_parameters():
            if any(opt_param in name for opt_param in args.optimize_params):
                param.requires_grad = True

    # Filter parameters based on optimize_params
    transformer_training_parameters = list(filter(lambda p: p.requires_grad, fusion_model.parameters()))
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size_local * accelerator.num_processes)
    print(len(transformer_training_parameters), args.learning_rate)

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_training_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    use_deepspeed_optimizer = (accelerator.state.deepspeed_plugin is not None) and ("optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config)
    use_deepspeed_scheduler = (accelerator.state.deepspeed_plugin is not None) and ("scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config)
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    """ ****************************  Accelerator setting.  **************************** """
    logger.info("=> Prepare everything with accelerator ...", main_process_only=True)
    fusion_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
        fusion_model, optimizer, train_dataloader, lr_scheduler 
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            else:
                accelerator.init_trackers(
                    project_name=args.wandb_init_args.project,
                    config=dict(args),
                    init_kwargs={
                        "wandb": {
                            "name": args.wandb_init_args.name,
                            "tags": args.wandb_init_args.tags.split(",") if isinstance(args.wandb_init_args.tags, str) else args.wandb_init_args.tags,
                            "dir": str(Path(args.output_dir)), 
                            "mode": args.wandb_init_args.mode,
                        }
                    }
                )
        else:
            tracker_name = args.tracker_name or "some-training"
            accelerator.init_trackers(tracker_name, config=vars(args))

    """ ****************************  Training info and resume.  **************************** """
    total_batch_size = args.train_batch_size_local * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size_local}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest": 
            dir_name = os.path.basename(args.resume_from_checkpoint)
            cur_full_path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(accelerator.project_dir, 'checkpoints'))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)
            dir_name = dirs[-1] if len(dirs) > 0 else None
            cur_full_path = os.path.join(accelerator.project_dir, 'checkpoints', dir_name)
        if dir_name is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {cur_full_path}")
            accelerator.load_state(cur_full_path)
            global_step = (int(dir_name.split('_')[-1]) + 1) * args.checkpointing_steps
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            iteration = int(global_step // args.checkpointing_steps)
            accelerator.project_configuration.iteration = iteration

    """ ****************************  Training.  **************************** """
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    for epoch in range(first_epoch, args.num_train_epochs):
        fusion_model.train()
        for step, batch in enumerate(train_dataloader):
            
            if global_step % args.validation.eval_steps == 0:
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
                    global_step=global_step
                )

            # TRAIN
            # TODO: Add ref model update
            models_to_accumulate = [fusion_model]
            with accelerator.accumulate(models_to_accumulate):
                loss_v, loss_a = training_step(batch, 
                                    fusion_model=fusion_model,
                                    # fusion_model_ref=fusion_model_ref,
                                    vae_model_video=vae_model_video,
                                    vae_model_audio=vae_model_audio,
                                    text_model=text_model,
                                    accelerator=accelerator, 
                                    dpo_beta=args.dpo_beta,
                                    load_dtype=weight_dtype)
                accelerator.backward(loss_v + loss_a)
                if accelerator.sync_gradients:
                    params_to_clip = fusion_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    checkpointing_step(model = fusion_model,
                                       accelerator = accelerator, 
                                       logger = logger, 
                                       args = args, 
                                       ckpt_idx = int(global_step // args.checkpointing_steps - 1))

            logs = {"loss": loss_v.detach().item() + loss_a.detach().item(),
                    "loss_v": loss_v.detach().item(), "loss_a": loss_a.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    logger.info("=> Saving the trained model ...")
    checkpointing_step(accelerator, logger, args)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = prepare_config()
    args, accelerator = prepare_everything(args)
    main(args, accelerator)
