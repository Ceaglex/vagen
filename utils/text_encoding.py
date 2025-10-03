import ftfy
import html
import regex as re
import torch
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import os
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def get_t5_prompt_embeds(
    prompt: Union[str, List[str]] = None,
    tokenizer = None,
    text_encoder = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
    return prompt_embeds



def encode_prompt(
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    tokenizer = None,
    text_encoder = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt,
            tokenizer = tokenizer,
            text_encoder = text_encoder,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = get_t5_prompt_embeds(
                prompt=negative_prompt,
                tokenizer = tokenizer,
                text_encoder = text_encoder,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds








def encode_prompt_sd(
    prompt,
    tokenizer,
    text_encoder,
    projection_model,
    device,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    negative_attention_mask: Optional[torch.LongTensor] = None,
):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # 1. Tokenize text
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) # type: ignore
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )

        text_input_ids = text_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 2. Text encoder forward
        text_encoder.eval()
        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    if do_classifier_free_guidance and negative_prompt is not None:
        uncond_tokens: List[str]
        if type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        # 1. Tokenize text
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        uncond_input_ids = uncond_input.input_ids.to(device)
        negative_attention_mask = uncond_input.attention_mask.to(device)

        # 2. Text encoder forward
        text_encoder.eval()
        negative_prompt_embeds = text_encoder(
            uncond_input_ids,
            attention_mask=negative_attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        if negative_attention_mask is not None:
            # set the masked tokens to the null embed
            negative_prompt_embeds = torch.where(
                negative_attention_mask.to(torch.bool).unsqueeze(2), negative_prompt_embeds, 0.0
            )

    # 3. Project prompt_embeds and negative_prompt_embeds
    if do_classifier_free_guidance and negative_prompt_embeds is not None:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if attention_mask is not None and negative_attention_mask is None:
            negative_attention_mask = torch.ones_like(attention_mask)
        elif attention_mask is None and negative_attention_mask is not None:
            attention_mask = torch.ones_like(negative_attention_mask)

        if attention_mask is not None:
            attention_mask = torch.cat([negative_attention_mask, attention_mask])

    prompt_embeds = projection_model(text_hidden_states=prompt_embeds).text_hidden_states
    if attention_mask is not None:
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

    if do_classifier_free_guidance:
        negative_prompt_embeds, prompt_embeds = prompt_embeds.chunk(2)
        return prompt_embeds, negative_prompt_embeds
    return prompt_embeds





def encode_duration_sd(
    projection_model,
    audio_start_in_s,
    audio_end_in_s,
    device,
    do_classifier_free_guidance,
    batch_size,
):
    audio_start_in_s = audio_start_in_s if isinstance(audio_start_in_s, list) else [audio_start_in_s]
    audio_end_in_s = audio_end_in_s if isinstance(audio_end_in_s, list) else [audio_end_in_s]

    if len(audio_start_in_s) == 1:
        audio_start_in_s = audio_start_in_s * batch_size
    if len(audio_end_in_s) == 1:
        audio_end_in_s = audio_end_in_s * batch_size

    # Cast the inputs to floats
    audio_start_in_s = [float(x) for x in audio_start_in_s]
    audio_start_in_s = torch.tensor(audio_start_in_s).to(device)

    audio_end_in_s = [float(x) for x in audio_end_in_s]
    audio_end_in_s = torch.tensor(audio_end_in_s).to(device)

    projection_output = projection_model(
        start_seconds=audio_start_in_s,
        end_seconds=audio_end_in_s,
    )
    seconds_start_hidden_states = projection_output.seconds_start_hidden_states
    seconds_end_hidden_states = projection_output.seconds_end_hidden_states

    # For classifier free guidance, we need to do two forward passes.
    # Here we repeat the audio hidden states to avoid doing two forward passes
    if do_classifier_free_guidance:
        seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
        seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states], dim=0)

    return seconds_start_hidden_states, seconds_end_hidden_states





def prepare_extra_step_kwargs(generator, eta, scheduler):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs





# @torch.no_grad()
# def inference(
#     transformer, 
#     vae,
#     tokenizer, 
#     text_encoder, 
#     weight_dtype, 
#     prompt_list,
#     negative_prompt = "",
#     negetive_prompt_embed = None,
#     num_videos_per_prompt = 1,
#     num_inference_steps = 50,
#     height = 480,
#     width = 832,
#     num_frames = 81,
#     guidance_scale = 5.0,
#     fps = 16,
#     output_dir = "./log",
#     device = 'cuda',
#     ):
#     # 解析 prompt

#     if negetive_prompt_embed is not None and type(negetive_prompt_embed) is str:
#         negetive_prompt_embed = torch.load(negetive_prompt_embed, map_location=device)
#     # elif negetive_prompt_embed is not None and type(negetive_prompt_embed) is torch.Tensor:
#     #     negetive_prompt_embed = negetive_prompt_embed.to(device)
#     os.makedirs(output_dir, exist_ok=True)

#     # 进入eval模式
#     transformer.eval()
#     text_encoder.eval()

#     for idx, prompt in enumerate(prompt_list):
#         # 文本编码
#         prompt_embeds = get_t5_prompt_embeds(
#             prompt=[prompt]*num_videos_per_prompt,
#             tokenizer=tokenizer,
#             text_encoder=text_encoder,
#             num_videos_per_prompt=num_videos_per_prompt,
#             max_sequence_length=512,
#             device=device,
#         ).to(dtype=weight_dtype)
#         if negetive_prompt_embed is not None:
#             negetive_prompt_embed = negetive_prompt_embed.to(dtype=weight_dtype, device=device)
#         else:
#             negetive_prompt_embed = get_t5_prompt_embeds(
#                 prompt=[negative_prompt]*num_videos_per_prompt,
#                 tokenizer=tokenizer,
#                 text_encoder=text_encoder,
#                 num_videos_per_prompt=num_videos_per_prompt,
#                 max_sequence_length=512,
#                 device=device,
#             ).to(dtype=weight_dtype)

#         # 采样初始噪声
#         batch_size = num_videos_per_prompt
#         vae_scale_factor_temporal = vae.config.scale_factor_temporal 
#         vae_scale_factor_spatial = vae.config.scale_factor_spatial 
#         num_channels_latents = transformer.config.in_channels 
#         num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
#         shape = (
#             batch_size,
#             num_channels_latents,
#             num_latent_frames,
#             int(height) // vae_scale_factor_spatial,
#             int(width) // vae_scale_factor_spatial,
#         )
#         latents = randn_tensor(shape, device=device, dtype=weight_dtype)

#         # 推理采样
#         scheduler = transformer.scheduler
#         scheduler.set_timesteps(num_inference_steps, device=device)
#         timesteps = scheduler.timesteps

#         for t in tqdm(timesteps, desc=f"Validation sampling prompt {idx}"):
#             latent_model_input = latents.to(weight_dtype)
#             timestep = t.expand(latents.shape[0])
#             with transformer.cache_context("cond"):
#                 noise_pred = transformer(
#                     hidden_states=latent_model_input,
#                     timestep=timestep,
#                     encoder_hidden_states=prompt_embeds,
#                     attention_kwargs=None,
#                     return_dict=False,
#                 )[0]
#             with transformer.cache_context("uncond"):
#                 noise_uncond = transformer(
#                     hidden_states=latent_model_input,
#                     timestep=timestep,
#                     encoder_hidden_states=negetive_prompt_embed,
#                     attention_kwargs=None,
#                     return_dict=False,
#                 )[0]
#             noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
#             latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

#         # 解码视频
#         latents = latents.to(vae.dtype)
#         latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
#         latents_std = 1 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
#         latents = latents / latents_std + latents_mean
#         video = vae.decode(latents, return_dict=False)[0]
#         # 后处理
#         if hasattr(transformer, "video_processor"):
#             video = transformer.video_processor.postprocess_video(video, output_type="np")
#         # 保存视频
#         for i in range(len(video)):
#             save_path = os.path.join(output_dir, f"val_prompt{idx}_sample{i}.mp4")
#             export_to_video(video[i], save_path, fps=fps)


#     print("[Validation] All prompts evaluated and saved.")

#     return None