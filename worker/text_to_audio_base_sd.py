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
from diffusers import AutoModel
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from worker.base import prepare_config, prepare_everything
from dataset.hy_video_audio import build_video_loader
from utils.audiomel import extract_batch_mel
from utils.text_encoding import get_t5_prompt_embeds, encode_prompt, encode_prompt_sd, encode_duration_sd, prepare_extra_step_kwargs
from utils.optimizer import get_optimizer, init_weights
from model.stable_audio.stable_audio_transformer import StableAudioDiTModel, StableAudioProjectionModel


# if is_wandb_available():
#     import wandb
#     wandb.init()
#     wandb.login(key='4aecda32c9818bce35ad00a8341aac5b1fb625ec')    
# check_min_version("0.35.0.dev0")
logger = get_logger(__name__)


def log_validation(
    config,
    vae,
    transformer,
    projection_model,
    tokenizer,
    text_encoder,
    step_scheduler,
    infer_dtype,
    accelerator,
    global_step
    ):

    with open(config.prompt_index_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = list(data.items())
    negative_prompt = config.negetive_prompt
    guidance_scale = config.guidance_scale
    do_classifier_free_guidance = guidance_scale > 1
    batch_size = 1
    device = accelerator.device
    output_dir = os.path.join(accelerator.project_dir, args.logging_subdir, str(global_step))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving path :{output_dir}")
    
    with torch.no_grad():

        generator = None         # generator = torch.Generator("cuda").manual_seed(0)
        num_channels_vae = transformer.module.config.in_channels if hasattr(transformer, "module") else transformer.config.in_channels
        waveform_length = int(config.audio_end_in_s * 22.5)
        # waveform_length = int(transformer.config.sample_size)
        shape = (batch_size * config.num_waveforms_per_prompt, num_channels_vae, waveform_length)

        
        idx = accelerator.process_index if accelerator.process_index < len(data) else 0
        step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
        data = data[idx::step_range]
        # print(accelerator.process_index, idx, prompt_list[idx::step_range])


        transformer.eval()
        for path, info in data:
            prompt = info['audio_caption']
            prompt_embeds, negative_prompt_embeds = encode_prompt_sd(
                prompt,
                tokenizer,
                text_encoder,
                projection_model,
                device,
                do_classifier_free_guidance,
                negative_prompt,
            ) # type: ignore



            with autocast(dtype=infer_dtype):
                latents = randn_tensor(shape, generator=generator, device=device, dtype=infer_dtype)
                rotary_embed_dim = transformer.module.config.attention_head_dim // 2 if hasattr(transformer, "module") else transformer.config.attention_head_dim // 2
                rotary_embedding = get_1d_rotary_pos_embed(
                    rotary_embed_dim,
                    latents.shape[2] + 1,
                    use_real=True,
                    repeat_interleave_real=False,
                )

                step_scheduler.set_timesteps(config.num_inference_steps, device=device)
                timesteps = step_scheduler.timesteps
                for i, t in tqdm(enumerate(timesteps)):
                    latent_model_input = latents
                    timestep = torch.stack([t for _ in range(latent_model_input.shape[0])])
                    noise_pred = transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds, 
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_uncond = transformer(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=negative_prompt_embeds, 
                            rotary_embedding=rotary_embedding,
                            return_dict=False,
                         )[0]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    latents = step_scheduler.step(noise_pred, t, latents).prev_sample

            # Check latent
            # latents = latents.reshape([batch_size * config.num_waveforms_per_prompt, -1, 8, latents.shape[-1]]).to(vae.dtype)  # [bs, 16, 8, latent_length]
            # latents = latents.transpose(-2,-3).transpose(-2, -1)                             # [bs, 8, latent_length, 16]
            # mel_spectrogram = vae.decode(latents).sample                                     # [num_waveforms_per_prompt, 1, num_latent_frames, n_mel(64)]
            # gen_audio = vocoder(mel_spectrogram.squeeze(1))                                  # [bs, duration*sr]
            gen_audio = vae.decode(latents.to(vae.dtype)).sample
            for i in range(len(gen_audio)):
                torchaudio.save(f"{output_dir}/{path.split('/')[-1][:-4]}.wav", gen_audio[i].cpu(), sample_rate=44100)
              
        transformer.train()




def training_step(batch, vae,
                  transformer, projection_model, tokenizer, text_encoder, 
                  accelerator, args):

    prompt = batch['prompt']  
    waveform = batch['audio_waveform']
    batch_size = len(prompt)

    with torch.no_grad():
        do_classifier_free_guidance = False
        prompt_embeds = encode_prompt_sd(
            prompt = prompt,
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            projection_model = projection_model,
            device = accelerator.device,
            do_classifier_free_guidance = do_classifier_free_guidance,
            negative_prompt = None,
        )

        latents = vae.encode(waveform.to(vae.dtype)).latent_dist.sample() # [bs, 64, 22.5 * duration]
        noise = torch.randn_like(latents)
        # # Check latent
        # gen_audio = vae.decode(latents.to(vae.dtype)).sample
        # for i in range(len(gen_audio)):
        #     torchaudio.save(f"test{i}.wav", gen_audio[i].cpu(), sample_rate=44100)

        timesteps = torch.randint( 0, 1000, (batch_size,), dtype=torch.int64, device=transformer.device).detach()
        rotary_embed_dim = transformer.module.config.attention_head_dim // 2 if hasattr(transformer, "module") else transformer.config.attention_head_dim // 2
        rotary_embedding = get_1d_rotary_pos_embed(
            rotary_embed_dim,
            latents.shape[2] + 1,
            use_real=True,
            repeat_interleave_real=False,
        )

        # NOTE SD-Flow follows Wan-2.1 takes `velocity = noise - latents` manner. Noise is at t=1.
        _t = (timesteps / 1000).to(latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
        noisy_latents = (1 - _t) * latents + _t * noise
        target = noise - latents

    model_pred = transformer(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds, 
        rotary_embedding=rotary_embedding,
    ).sample
    loss = nn_func.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def checkpointing_step(accelerator, logger, args):
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

    # os.makedirs(save_path, exist_ok=True)
    accelerator.save_state()
    logger.info(f"Saved state to {ckpt_dir}")


def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.float32   # 推理时候用torch.float16 更快


    transformer = StableAudioDiTModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype = load_dtype,
        local_files_only=True,                  # From pretrained
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True,
        use_safetensors=True,                          
    )

    projection_model = StableAudioProjectionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="projection_model",
        torch_dtype=load_dtype, 
        local_files_only=True,
        use_safetensors=True,
    )

    vae = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path,         
        subfolder="vae", 
        torch_dtype = load_dtype,
        local_files_only=True,
        use_safetensors=True,                      
    )

    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        torch_dtype=load_dtype, 
        local_files_only=True,
        use_safetensors=True,
    )

    tokenizer = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        local_files_only=True,
        use_safetensors=True,
    )


    # transformer.proj_in.apply(init_weights)
    # transformer.proj_in.apply(init_weights)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vae.eval()
    vae.requires_grad_(False)


    """ ****************************  Helper function.  **************************** """
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    """ ****************************  Weights dtype setting.  **************************** """
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    infer_dtype = torch.float16
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
        # Only upcast trainable parameters (LoRA) into fp32.
        cast_training_params([transformer], dtype=torch.float32)

    """ ****************************  Data setting.  **************************** """
    logger.info("=> Preparing training data...", main_process_only=True)
    dataset_cfg = args.hy_dataloader
    train_dataloader = build_video_loader(args=dataset_cfg)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size_local

    """ ****************************  Optimization setting.  **************************** """
    transformer_training_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    
    if args.scale_lr:
        args.learning_rate = ( args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size_local * accelerator.num_processes )
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_training_parameters, "lr": args.learning_rate}
    print( args.learning_rate)
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
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
        transformer, optimizer, train_dataloader, lr_scheduler 
    )
    projection_model = projection_model.to(accelerator.device)
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
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
            # step_file = [f for f in os.listdir(cur_full_path) if f.startswith("global-step-is-")]
            # global_step = int(step_file[0].split("-")[-1])     # NOTE Assume there is only one step file

            global_step = (int(dir_name.split('_')[-1]) + 1) * args.checkpointing_steps
            global_step = 0 ####
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch



    """ ****************************  Training.  **************************** """
    # Only show the progress bar once on each machine.
    progress_bar = tqdm( range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process, )
    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            
            if global_step % args.validation.eval_steps == 0:
                # TODO: Scheduler
                # step_scheduler = AutoModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", )
                step_scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=args.validation.flow_shift)         

                log_validation(
                    config = args.validation,
                    vae = vae, 
                    transformer = transformer,
                    projection_model = projection_model,
                    tokenizer = tokenizer,
                    text_encoder = text_encoder,
                    step_scheduler = step_scheduler,
                    infer_dtype = infer_dtype,
                    accelerator = accelerator,
                    global_step = global_step,
                )
                
                del step_scheduler
                torch.cuda.empty_cache()
                gc.collect()
                free_memory()
            

            # TRAIN
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                loss = training_step(batch, vae, 
                                    #  vocoder,
                                     transformer, projection_model, tokenizer, text_encoder, 
                                    #  weight_dtype, 
                                     accelerator, args)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
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


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
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