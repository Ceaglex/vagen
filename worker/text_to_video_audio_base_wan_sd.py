# import pdb; pdb.set_trace()
import logging, math, os, shutil
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import copy
import soundfile as sf
import json
# from datetime import datetime
# from typing import List, Optional, Tuple, Union, Dict, Any
import time

import torch
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
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from peft import LoraConfig, get_peft_model

from worker.base import prepare_config, prepare_everything
from dataset.hy_video_audio import build_video_loader
from utils.va_processing import add_audio_to_video
from utils.text_encoding import get_t5_prompt_embeds, encode_prompt, encode_prompt_sd, encode_duration_sd, prepare_extra_step_kwargs
from utils.optimizer import get_optimizer, init_weights, init_class, load_weights
from model.stable_audio.stable_audio_transformer import StableAudioDiTModel, StableAudioProjectionModel
from model.jointva.jointva_transformer import JointVADiTModel
from model.wan.pipeline_wan_ttv import AutoencoderKLWan

# if is_wandb_available():
#     import wandb
#     wandb.init()
#     wandb.login(key='4aecda32c9818bce35ad00a8341aac5b1fb625ec')    
# check_min_version("0.35.0.dev0")
logger = get_logger(__name__)



def log_validation(
    config,
    vajoint_dit,

    audio_vae,
    audio_projection_model,
    audio_tokenizer,
    audio_text_encoder,

    video_vae,
    video_processor,
    video_text_encoder,
    video_tokenizer,

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
    a_step_scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=config.scheduler.num_train_timesteps, flow_shift=config.scheduler.flow_shift)         
    v_step_scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=config.scheduler.num_train_timesteps, flow_shift=config.scheduler.flow_shift)         

    num_va_per_prompt = config.num_va_per_prompt
    num_inference_steps = config.num_inference_steps
    v_negative_prompt = config.v_negative_prompt
    v_num_frames = config.v_num_frames
    v_height = config.v_height
    v_width = config.v_width
    v_guidance_scale = config.v_guidance_scale
    a_negative_prompt = config.a_negative_prompt
    a_duration = config.a_duration
    a_guidance_scale = config.a_guidance_scale
    do_classifier_free_guidance = (a_guidance_scale > 1.0) or (v_guidance_scale > 1.0)


    vajoint_dit.eval()
    # vajoint_dit.eval()
    with torch.no_grad():
        with autocast(dtype=infer_dtype):
            torch.cuda.empty_cache()
            gc.collect()
            free_memory()
            for path, info in data:
                v_prompt = info['video_caption']
                a_prompt = info['audio_caption']

                # Prepare shapes
                a_num_channels_vae = vajoint_dit.module.audio_transformer.config.in_channels if hasattr(vajoint_dit, "module") else vajoint_dit.audio_transformer.config.in_channels
                waveform_length = int(a_duration * 21.5)
                a_shape = (num_va_per_prompt, a_num_channels_vae, waveform_length)
                
                v_num_channels_vae = vajoint_dit.module.video_transformer.config.in_channels if hasattr(vajoint_dit, "module") else vajoint_dit.video_transformer.config.in_channels
                num_latent_frames = (v_num_frames - 1) // video_vae.config.scale_factor_temporal + 1
                v_shape = (
                    num_va_per_prompt,
                    v_num_channels_vae,
                    num_latent_frames,
                    v_height // video_vae.config.scale_factor_spatial,
                    v_width // video_vae.config.scale_factor_spatial,
                )


                # Prepare Condition
                v_prompt_embeds, v_negative_prompt_embeds = encode_prompt(
                    prompt=v_prompt,
                    negative_prompt=v_negative_prompt,
                    tokenizer = video_tokenizer,
                    text_encoder = video_text_encoder,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    num_videos_per_prompt=num_va_per_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    max_sequence_length=512,
                    device=accelerator.device,
                    dtype=infer_dtype, # 必须是float32 / bfloat16，float16有问题，不知道为何
                ) # type: ignore
                a_prompt_embeds, a_negative_prompt_embeds = encode_prompt_sd(
                    a_prompt,
                    audio_tokenizer,
                    audio_text_encoder,
                    audio_projection_model,
                    accelerator.device,
                    do_classifier_free_guidance,
                    a_negative_prompt,
                ) # type: ignore
                a_rotary_embed_dim = vajoint_dit.module.audio_transformer.config.attention_head_dim // 2 if hasattr(vajoint_dit, "module") else vajoint_dit.audio_transformer.config.attention_head_dim // 2
                a_rotary_embedding = get_1d_rotary_pos_embed(a_rotary_embed_dim, a_shape[2] + 1, use_real=True, repeat_interleave_real=False,)


                # Initialize latents
                a_latents = randn_tensor(a_shape, generator=a_generator, device=accelerator.device, dtype=infer_dtype)
                v_latents = randn_tensor(v_shape, generator=v_generator, device=accelerator.device, dtype=infer_dtype)
                v_step_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
                a_step_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
                timesteps = v_step_scheduler.timesteps
                    
                    
                # Denoising loop
                for i, t in tqdm(enumerate(timesteps)):
                    timestep = t.expand(num_va_per_prompt)
                    v_latent_model_input = v_latents
                    a_latent_model_input = a_latents

                    # Forward pass through joint model
                    v_noise_pred, a_noise_pred = vajoint_dit(
                        timestep=timestep,
                        v_latent_model_input=v_latent_model_input,
                        v_encoder_hidden_states=v_prompt_embeds,
                        v_attention_kwargs=None,
                        a_latent_model_input=a_latent_model_input,
                        a_encoder_hidden_states=a_prompt_embeds, 
                        a_rotary_embedding=a_rotary_embedding,
                        return_dict=False)
                    v_noise_pred = v_noise_pred[0].to(infer_dtype)
                    a_noise_pred = a_noise_pred[0].to(infer_dtype)

                    if do_classifier_free_guidance:
                        v_noise_uncond, a_noise_uncond = vajoint_dit(
                            timestep=timestep,
                            v_latent_model_input=v_latent_model_input,
                            v_encoder_hidden_states=v_negative_prompt_embeds,
                            v_attention_kwargs=None,
                            a_latent_model_input=a_latent_model_input,
                            a_encoder_hidden_states=a_negative_prompt_embeds, 
                            a_rotary_embedding=a_rotary_embedding,
                            return_dict=False)
                        v_noise_uncond = v_noise_uncond[0].to(infer_dtype)
                        a_noise_uncond = a_noise_uncond[0].to(infer_dtype)
                        v_noise_pred = v_noise_uncond + v_guidance_scale * (v_noise_pred - v_noise_uncond)
                        a_noise_pred = a_noise_uncond + a_guidance_scale * (a_noise_pred - a_noise_uncond)

                    v_latents = v_step_scheduler.step(v_noise_pred, t, v_latents, return_dict=False)[0]
                    a_latents = a_step_scheduler.step(a_noise_pred, t, a_latents, return_dict=False)[0] 

                # Decode latents to final outputs
                v_latents = v_latents.to(video_vae.dtype)
                v_latents_mean = torch.tensor(video_vae.config.latents_mean).view(1, video_vae.config.z_dim, 1, 1, 1).to(accelerator.device, video_vae.dtype)
                v_latents_std = 1 / torch.tensor(video_vae.config.latents_std).view(1, video_vae.config.z_dim, 1, 1, 1).to(accelerator.device, video_vae.dtype)
                v_latents = v_latents / v_latents_std + v_latents_mean
                    
                gen_video = video_vae.decode(v_latents, return_dict=False)[0]
                gen_video = video_processor.postprocess_video(gen_video.to(infer_dtype), output_type='np')
                gen_audio = audio_vae.decode(a_latents.to(audio_vae.dtype)).sample

                # Save results. BS and num_va_per_prompt = 1
                for i in range(num_va_per_prompt):
                    v_path = f"{output_dir}/{path.split('/')[-1][:-4]}.mp4"
                    a_path = f"{output_dir}/{path.split('/')[-1][:-4]}.wav"
                    o_path = f"{output_dir}/{path.split('/')[-1][:-4]}_.mp4"
                    export_to_video(gen_video[i], v_path, fps=config.data_info.video_info.fps)
                    torchaudio.save(a_path, gen_audio[i].cpu().to(torch.float32), sample_rate=config.data_info.audio_info.sr)
                    add_audio_to_video(video_path = v_path, audio_path = a_path, output_path = o_path)
                    
            # Clear cache after validation
            torch.cuda.empty_cache()
            gc.collect()
            free_memory()
    vajoint_dit.train()


def training_step(batch, 
                  vajoint_dit,
                  audio_vae,
                  audio_projection_model,
                  audio_tokenizer,
                  audio_text_encoder,
                  video_vae,
                  video_processor,
                  video_text_encoder,
                  video_tokenizer, 
                  accelerator, 
                  load_dtype):

    v_prompt = batch['v_prompt']
    a_prompt = batch['a_prompt']  
    waveform = batch['audio_waveform']
    video_pixel = batch['video_pixel']
    batch_size = len(v_prompt)

    
    # with autocast(dtype=load_dtype):
    with torch.no_grad():
        a_prompt_embeds = encode_prompt_sd(
            prompt = a_prompt,
            tokenizer = audio_tokenizer,
            text_encoder = audio_text_encoder,
            projection_model = audio_projection_model,
            device = accelerator.device,
            do_classifier_free_guidance = False,
            negative_prompt = None,
        )
        v_prompt_embeds = get_t5_prompt_embeds(
            prompt=v_prompt,
            tokenizer = video_tokenizer,
            text_encoder = video_text_encoder,
            device=accelerator.device,
        )

        # TODO: Faster Video VAE
        v_latents = video_vae.encode(video_pixel.to(video_vae.dtype)).latent_dist.sample()
        v_latents_mean = torch.tensor(video_vae.config.latents_mean).view(1, video_vae.config.z_dim, 1, 1, 1).to(v_latents.device, v_latents.dtype)
        v_latents_std = 1.0 / torch.tensor(video_vae.config.latents_std).view(1, video_vae.config.z_dim, 1, 1, 1).to(v_latents.device, v_latents.dtype)
        v_latents = (v_latents - v_latents_mean) * v_latents_std
        v_noise = torch.randn_like(v_latents)
        a_latents = audio_vae.encode(waveform.to(audio_vae.dtype)).latent_dist.sample() # [bs, 64, 21.5 * duration]
        a_noise = torch.randn_like(a_latents)
        a_rotary_embed_dim = vajoint_dit.module.audio_transformer.config.attention_head_dim // 2 if hasattr(vajoint_dit, "module") else vajoint_dit.audio_transformer.config.attention_head_dim // 2
        a_rotary_embedding = get_1d_rotary_pos_embed(a_rotary_embed_dim, a_latents.shape[2] + 1, use_real=True, repeat_interleave_real=False)
        # a_rotary_embedding = (a_rotary_embedding[0].to(dtype=load_dtype), a_rotary_embedding[1].to(dtype=load_dtype))
        # # # # TODO : Check latent
        # # # Decode latents to final outputs
        # # v_latents = v_latents.to(video_vae.dtype)
        # # v_latents_mean = torch.tensor(video_vae.config.latents_mean).view(1, video_vae.config.z_dim, 1, 1, 1).to(v_latents.device, v_latents.dtype)
        # # v_latents_std = 1 / torch.tensor(video_vae.config.latents_std).view(1, video_vae.config.z_dim, 1, 1, 1).to(v_latents.device, v_latents.dtype)
        # # v_latents = v_latents / v_latents_std + v_latents_mean      
        # # gen_audio = audio_vae.decode(a_latents.to(audio_vae.dtype)).sample
        # # gen_video = video_vae.decode(v_latents.to(video_vae.dtype)).sample
        # # gen_video = video_processor.postprocess_video(gen_video, output_type='np')
        # # for i in range(len(gen_video)):
        # #     export_to_video(gen_video[i], f"test{i}.mp4", fps=16)
        # #     torchaudio.save(f"test{i}.wav", gen_audio[i].to(torch.float32).cpu(), sample_rate=44100)

        timesteps = torch.randint( 0, 1000, (batch_size,), dtype=torch.int64, device=vajoint_dit.device).detach()
        a_t = (timesteps / 1000).to(a_latents.dtype).view(-1, *([1] * (a_latents.ndim - 1)))
        v_t = (timesteps / 1000).to(v_latents.dtype).view(-1, *([1] * (v_latents.ndim - 1)))
        a_latent_model_input = (1 - a_t) * a_latents + a_t * a_noise
        v_latent_model_input = (1 - v_t) * v_latents + v_t * v_noise
        a_target = a_noise - a_latents
        v_target = v_noise - v_latents


    v_noise_pred, a_noise_pred = vajoint_dit(
        timestep=timesteps,
        v_latent_model_input=v_latent_model_input,
        v_encoder_hidden_states=v_prompt_embeds,
        v_attention_kwargs=None,
        a_latent_model_input=a_latent_model_input,
        a_encoder_hidden_states=a_prompt_embeds, 
        a_rotary_embedding=a_rotary_embedding,
        return_dict=False
    )
    v_noise_pred = v_noise_pred[0]
    a_noise_pred = a_noise_pred[0]

    loss_v = nn_func.mse_loss(v_noise_pred.float(), v_target.float(), reduction="mean")
    loss_a = nn_func.mse_loss(a_noise_pred.float(), a_target.float(), reduction="mean")
    return loss_v, loss_a


def checkpointing_step(accelerator, logger, args, ckpt_idx = 0):
    # ckpt_dir = os.path.join(args.output_dir, args.ckpt_subdir)
    ckpt_dir = os.path.join(accelerator.project_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(ckpt_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info( f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints" )
            logger.info( f"Removing checkpoints: {', '.join(removing_checkpoints)}" )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    # out_path = f"{ckpt_dir}/checkpoints_{ckpt_idx}"
    accelerator.save_state()
    logger.info(f"Saved state to {ckpt_dir}")


def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32  # TODO: Set Correct Dtype
    infer_dtype = torch.bfloat16

    # JointVADiT
    joint_va = JointVADiTModel(
        load_dtype = load_dtype,
        audio_transformer_path = args.audio_pretrained_model_name_or_path, 
        audio_transformer_weights_path = args.audio_transformer_safetensors_path,
        video_transformer_path = args.video_pretrained_model_name_or_path,
        video_transformer_weights_path = args.video_transformer_safetensors_path,
        bridge_config = args.bridge_config,
        bridge_weights_path = args.bridge_safetensors_path
    )

    # Audio Model Components
    audio_vae = init_class(AutoModel, load_dtype, args.audio_pretrained_model_name_or_path, subfolder="vae", set_eval=True)
    audio_projection_model = init_class(StableAudioProjectionModel, load_dtype, args.audio_pretrained_model_name_or_path, subfolder="projection_model", set_eval=True)
    audio_text_encoder = init_class(AutoModel, load_dtype, args.audio_pretrained_model_name_or_path, subfolder="text_encoder", set_eval=True)
    audio_tokenizer = init_class(AutoModel, load_dtype, args.audio_pretrained_model_name_or_path, subfolder="tokenizer")

    # Video Model Components
    video_vae = init_class(AutoencoderKLWan, load_dtype, args.video_pretrained_model_name_or_path, subfolder="vae", set_eval=True)
    video_processor = VideoProcessor(vae_scale_factor=video_vae.config.scale_factor_spatial) if video_vae is not None else None
    video_text_encoder = init_class(AutoModel, load_dtype, args.video_pretrained_model_name_or_path, subfolder="text_encoder", set_eval=True)
    video_tokenizer = init_class(AutoModel, load_dtype, args.video_pretrained_model_name_or_path, subfolder="tokenizer")


    # Set Training Parameters
    if args.gradient_checkpointing:
        joint_va.enable_gradient_checkpointing()



    """ ****************************  Weights dtype setting.  **************************** """
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = load_dtype
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
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
    # Due to pytorch#99272, MPS does not yet support bfloat16.
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([joint_va], dtype=torch.float32)


    """ ****************************  Data setting.  **************************** """
    logger.info("=> Preparing training data...", main_process_only=True)
    dataset_cfg = args.hy_dataloader
    train_dataloader = build_video_loader(args=dataset_cfg)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size_local


    """ ****************************  Optimization setting.  **************************** """
    joint_va.requires_grad_(False)
    if args.lora_config.use_lora == True:
        lora_config = LoraConfig(
            r=args.lora_config.rank,
            lora_alpha=args.lora_config.lora_alpha,
            target_modules=args.lora_config.lora_modules,
            lora_dropout=0.1,
            bias="none",)
        joint_va.audio_transformer = get_peft_model(joint_va.audio_transformer, lora_config)  # type: ignore[arg-type]
        joint_va.video_transformer = get_peft_model(joint_va.video_transformer, lora_config)  # type: ignore[arg-type]
    if args.block_config.train_va == True:
        for block_idx in args.block_config.audio_trainable_blocks:
            joint_va.audio_transformer.transformer_blocks[block_idx].requires_grad_(True)
        for block_idx in args.block_config.video_trainable_blocks:
            joint_va.video_transformer.blocks[block_idx].requires_grad_(True)
    if args.optimize_params is not None:
        for name, param in joint_va.named_parameters():
            if any(opt_param in name for opt_param in args.optimize_params):
                param.requires_grad = True

    # Filter parameters based on optimize_params
    transformer_training_parameters = list(filter(lambda p: p.requires_grad, joint_va.parameters()))
    if args.scale_lr:
        args.learning_rate = ( args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size_local * accelerator.num_processes )
    print(len(transformer_training_parameters), args.learning_rate)

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_training_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    use_deepspeed_optimizer = ( accelerator.state.deepspeed_plugin is not None ) and ( "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config )
    use_deepspeed_scheduler = ( accelerator.state.deepspeed_plugin is not None ) and ( "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config )
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
    joint_va, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
        joint_va, optimizer, train_dataloader, lr_scheduler 
    )
    audio_projection_model = audio_projection_model.to(accelerator.device)
    audio_vae = audio_vae.to(accelerator.device)
    audio_text_encoder = audio_text_encoder.to(accelerator.device)
    video_vae = video_vae.to(accelerator.device)
    video_text_encoder = video_text_encoder.to(accelerator.device)
    # print("\n==================================================\n")
    # for name, param in joint_va.named_parameters():
    #     if not param.requires_grad:
    #         print(name)
    # print("\n==================================================\n")


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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
    # logger.info(f"  Num examples = {len(train_dataset)}")
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
            # Get the mos recent checkpoint
            # dirs = os.listdir(os.path.join(args.output_dir, args.ckpt_subdir))
            dirs = os.listdir(os.path.join(accelerator.project_dir, 'checkpoints'))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)
            dir_name = dirs[-1] if len(dirs) > 0 else None
            cur_full_path = os.path.join(accelerator.project_dir, 'checkpoints', dir_name)
        if dir_name is None:
            accelerator.print( f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run." )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print( f"Resuming from checkpoint {cur_full_path}" )
            accelerator.load_state(cur_full_path)
            global_step = (int(dir_name.split('_')[-1]) + 1) * args.checkpointing_steps
            # global_step = 0 ####
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            iteration = int(global_step // args.checkpointing_steps)
            accelerator.project_configuration.iteration = iteration
            



    """ ****************************  Training.  **************************** """
    # Only show the progress bar once on each machine.
    progress_bar = tqdm( range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process, )
    for epoch in range(first_epoch, args.num_train_epochs):
        joint_va.train()
        for step, batch in enumerate(train_dataloader):
            
            if global_step % args.validation.eval_steps == 0:
                log_validation(
                    config = args.validation,
                    vajoint_dit = joint_va,
                    audio_vae = audio_vae,
                    audio_projection_model = audio_projection_model,
                    audio_tokenizer = audio_tokenizer,
                    audio_text_encoder = audio_text_encoder,
                    video_vae = video_vae,
                    video_processor = video_processor,
                    video_text_encoder = video_text_encoder,
                    video_tokenizer = video_tokenizer,
                    infer_dtype = infer_dtype,
                    accelerator = accelerator,
                    global_step = global_step
                )


            # TRAIN
            models_to_accumulate = [joint_va]
            with accelerator.accumulate(models_to_accumulate):
                loss_v, loss_a = training_step(batch, 
                                    vajoint_dit = joint_va,
                                    audio_vae = audio_vae,
                                    audio_projection_model = audio_projection_model,
                                    audio_tokenizer = audio_tokenizer,
                                    audio_text_encoder = audio_text_encoder,
                                    video_vae = video_vae,
                                    video_processor = video_processor,
                                    video_text_encoder = video_text_encoder,
                                    video_tokenizer = video_tokenizer,
                                    accelerator = accelerator, 
                                    load_dtype = weight_dtype)
                accelerator.backward(loss_v + loss_a)
                if accelerator.sync_gradients:
                    params_to_clip = joint_va.parameters()
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
                    checkpointing_step(accelerator, logger, args)


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
