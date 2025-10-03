# import pdb; pdb.set_trace()
# 这份代码好久之前debug用过，不过很久没改过了，有时间可以照着text_to_audio_base_sd.py修改一下
import logging, math, os, shutil
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import copy
# from datetime import datetime
# from typing import List, Optional, Tuple, Union, Dict, Any

import torch
# import torch.nn as nn
import torch.nn.functional as nn_func
from torch.cuda.amp import autocast
# from torch.utils.data import DataLoader, Dataset
# import transformers
import gc
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
# from huggingface_hub import create_repo, upload_folder
# from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# from diffusers import WanPipeline
from diffusers import AutoModel

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor

# from model.wan.wan_transformer_for_audio import WanTransformer3DModel
from model.wan.wan_transformer_for_video import WanTransformer3DModel
from model.wan.pipeline_wan_ttv import FlowMatchEulerDiscreteScheduler, AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor

from worker.base import prepare_config, prepare_everything
from dataset.hy_video_audio import build_video_loader
from utils.text_encoding import get_t5_prompt_embeds, encode_prompt
from utils.optimizer import get_optimizer


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
    tokenizer,
    text_encoder,
    scheduler,
    video_processor,
    load_dtype,
    infer_dtype,
    device,
    ):

    with torch.no_grad():
        prompt_list = config.prompt.split(config.prompt_separator)
        negative_prompt = config.negetive_prompt
        output_dir = getattr(config, "save_dir", 5.0)
        os.makedirs(output_dir, exist_ok=True)
        vae.to(device)
        transformer.eval()

        for prompt_idx, prompt in enumerate(prompt_list):

            torch.cuda.empty_cache()
            gc.collect()
            free_memory()
            
            prompt_embeds, negative_prompt_embeds = None, None
            if config.negetive_prompt_embed is not None:
                negative_prompt_embeds = torch.load(config.negetive_prompt_embed).to(device).unsqueeze(0)
            prompt_embeds, negative_prompt_embeds = encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                tokenizer = tokenizer,
                text_encoder = text_encoder,
                do_classifier_free_guidance=config.guidance_scale > 1.0,
                num_videos_per_prompt=config.num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=512,
                device=device,
            ) # type: ignore

            num_latent_frames = (config.num_frames - 1) // vae.config.scale_factor_temporal + 1
            num_channels = transformer.module.config.in_channels if hasattr(transformer, "module") else transformer.config.in_channels
            shape = (
                config.num_videos_per_prompt,
                num_channels,
                num_latent_frames,
                config.height // vae.config.scale_factor_spatial,
                config.width // vae.config.scale_factor_spatial,
            )


            with autocast(dtype=infer_dtype):
                
                prompt_embeds = prompt_embeds.to(infer_dtype)
                if negative_prompt_embeds is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(infer_dtype)
                latents = randn_tensor(shape, device=device, dtype=infer_dtype)
                scheduler.set_timesteps(config.num_inference_steps, device=device)
                timesteps = scheduler.timesteps
                # current_model = transformer.module if hasattr(transformer, "module") else transformer

                for i, t in tqdm(enumerate(timesteps)):

                    torch.cuda.empty_cache()
                    gc.collect()
                    free_memory()

                    current_guidance_scale = config.guidance_scale
                    latent_model_input = latents
                    timestep = t.expand(latents.shape[0])

                    noise_pred = transformer.module(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    if config.guidance_scale > 1.0:
                        noise_uncond = transformer.module(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    
                latents = latents.to(vae.dtype)
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
                latents_std = 1 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
                latents = latents / latents_std + latents_mean
                video = vae.decode(latents, return_dict=False)[0]
                video = video_processor.postprocess_video(video.to(load_dtype), output_type='np')
                for i in range(config.num_videos_per_prompt):
                    export_to_video(video[i], f"{output_dir}/output{prompt_idx}_{i}.mp4", fps=config.fps if hasattr(config, "fps") else 16)


                del timesteps,  timestep, prompt_embeds, negative_prompt_embeds, latent_model_input, latents, video, noise_pred, noise_uncond
                torch.cuda.empty_cache()
                gc.collect()
                free_memory()

        vae.to('cpu')
        transformer.train()




def training_step(batch, transformer, tokenizer, text_encoder, weight_dtype, accelerator):
    # Video latents [1, 16, 33, 44, 78], Audio latents [1, 128, 400]
    prompt = batch['prompt']  
    latents = batch['video_latent'].to(accelerator.device, dtype=weight_dtype)

    # Encode prompts.
    with torch.no_grad():
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt,
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            device=accelerator.device,
        )
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        prompt_embeds = prompt_embeds.detach()

    # Sample noise that will be added to the latents
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]

    # Sample a random timestep for each image. TODO Check a better way than specific 1000.
    timesteps = torch.randint( 0, 1000, (batch_size,), ).to(accelerator.device)
    timesteps = timesteps.long()

    # NOTE Wan-2.1 takes `velocity = noise - latents` manner. Noise is at t=1.
    _t = (timesteps / 1000).to(latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
    noisy_latents = (1 - _t) * latents + _t * noise
    target = noise - latents
    noisy_latents = noisy_latents.to(weight_dtype)

    model_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        attention_kwargs=None,
    ).sample
    loss = nn_func.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def checkpointing_step(accelerator, logger, args, global_step, epoch, is_final_checkpoint: bool = False):
    ckpt_dir = os.path.join(args.output_dir, args.ckpt_subdir)
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

    final_tag = "-final" if is_final_checkpoint else ""
    save_path = os.path.join(ckpt_dir, f"checkpoint{final_tag}-epoch{epoch:04d}-step{global_step:.3e}")
    accelerator.save_state(save_path)
    step_file = os.path.join(save_path, f"global-step-is-{str(global_step)}")
    open(step_file, 'a').close()
    logger.info(f"Saved state to {save_path}")


def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.float32   # 推理时候用torch.float16 更快

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", 
        torch_dtype=load_dtype
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", 
        torch_dtype=load_dtype, 
        local_files_only=True,
        low_cpu_mem_usage=False, 
        use_safetensors=True,
        ignore_mismatched_sizes=True,      # Setting for model structure changes
    )
    tokenizer = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", 
        local_files_only=True,
        use_safetensors=True,
    )
    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", 
        torch_dtype=load_dtype, 
        local_files_only=True,
        use_safetensors=True,
    )
    
    
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to('cpu')




    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

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
    # Prepare everything with our `accelerator`.
    # transformer, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
    #     transformer, text_encoder, optimizer, train_dataloader, lr_scheduler 
    # )
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
        transformer, optimizer, train_dataloader, lr_scheduler 
    )
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
                            "dir": str(Path(args.output_dir, args.wandb_subdir)), 
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
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(os.path.join(args.output_dir, args.ckpt_subdir))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)
            dir_name = dirs[-1] if len(dirs) > 0 else None

        if dir_name is None:
            accelerator.print( f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run." )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print( f"Resuming from checkpoint {dir_name}" )
            cur_full_path = os.path.join(args.output_dir, args.ckpt_subdir, dir_name)
            accelerator.load_state(cur_full_path)
            step_file = [f for f in os.listdir(cur_full_path) if f.startswith("global-step-is-")]
            global_step = int(step_file[0].split("-")[-1])     # NOTE Assume there is only one step file

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
            
            # INFERENCE
            if (epoch * len(train_dataloader) + step) % args.validation.eval_steps == 0:
                scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=args.validation.flow_shift)
                video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
                # print(f"infer_dtype: {infer_dtype} {text_encoder.dtype}, load_dtype: {load_dtype}, weight_dtype: {weight_dtype}")

                log_validation(
                    config = args.validation,
                    vae = vae, 
                    transformer = transformer,
                    tokenizer = tokenizer,
                    text_encoder = text_encoder,
                    scheduler = scheduler,
                    video_processor = video_processor,
                    load_dtype = load_dtype,
                    infer_dtype = infer_dtype,
                    device = accelerator.device,
                )
                
                del scheduler, video_processor
                torch.cuda.empty_cache()
                gc.collect()
                free_memory()
            
            # TRAIN
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                loss = training_step(batch, transformer, tokenizer, text_encoder, weight_dtype, accelerator)
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

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        checkpointing_step(accelerator, logger, args, global_step, epoch,)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("=> Saving the trained model ...")
        checkpointing_step(accelerator, logger, args, global_step, epoch, is_final_checkpoint=True)

        # Cleanup trained models to save memory
        del transformer
        free_memory()

        # Final test inference
        # TODO

        # NOTE This version does not support saving to hub.
        # if args.push_to_hub:  # save and upload to hub.
            
    accelerator.end_training()


if __name__ == "__main__":
    args = prepare_config()
    args, accelerator = prepare_everything(args)
    main(args, accelerator)