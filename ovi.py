import pdb; pdb.set_trace()
import torch
import os
from tqdm import tqdm
from utils.fm_solvers import FlowUniPCMultistepScheduler
from utils.va_processing import snap_hw_to_multiple_of_32, save_video
from utils.model_loading import (
    init_fusion_score_model_ovi, 
    init_text_model, 
    init_mmaudio_vae, 
    init_wan_vae_2_2, 
    load_fusion_checkpoint
)


device = 0
target_dtype = torch.bfloat16
ckpt_dir = "/home/chengxin/chengxin/Ovi/ckpts"  # 请修改为你的检查点路径

model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=True)
if ckpt_dir is not None:
    checkpoint_path = os.path.join(ckpt_dir, "Ovi", "model.safetensors")
    load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=True)
model = model.requires_grad_(False).eval().to(dtype=target_dtype).to(device=device)
model.set_rope_params()



vae_model_video = init_wan_vae_2_2(ckpt_dir, rank=device)
vae_model_video.model.requires_grad_(False).eval().to(target_dtype)
vae_model_audio = init_mmaudio_vae(ckpt_dir, rank=device)
vae_model_audio.requires_grad_(False).eval().to(target_dtype)


text_model = init_text_model(ckpt_dir, rank=device, cpu_offload=False)
image_model = None


audio_latent_channel = audio_config.get("in_dim")
video_latent_channel = video_config.get("in_dim")
audio_latent_length = 157
video_latent_length = 31






sample_steps = 10
shift = 5.0
text_prompt = "In a cozy indoor setting with warm, soft lighting that casts gentle shadows across the wooden floor, a fawn-colored Boxer dog with white markings on its paws and chest lies sprawled comfortably on a textured beige rug. The atmosphere is calm and inviting, with the dog exuding a sense of relaxed contentment. The dog's sleek coat gleams subtly under the ambient light, and its expressive brown eyes convey a mix of curiosity and playfulness. As the camera maintains a steady, eye-level shot, the dog begins to stretch languidly, extending its front legs forward and arching its back slightly, causing its muscles to ripple beneath its skin. Its ears twitch gently, and its tail wags slowly, creating a subtle breeze that ruffles the fine hairs on its body. The dog then shifts its weight, rolling slightly onto its side, its paws moving in a playful manner as if mimicking running motions. The interaction between the dog's paws and the rug creates a slight friction, visible as the fibers of the rug compress and release with each movement. In the background, a wooden cabinet with a soft blanket draped over it adds to the homely ambiance, while the polished wooden floor reflects the warm tones of the room. The entire scene is captured in a photorealistic, highly detailed 8K quality, emphasizing the textures and natural movements of the dog within this serene domestic environment. <AUDCAP>The gentle sounds of a dog barking softly fill the calm and inviting indoor atmosphere, blending seamlessly with the subtle rustling of the dog's paws against the textured rug.<ENDAUDCAP>"
video_negative_prompt = 'jitter, bad hands, blur, distortion'
audio_negative_prompt = 'robotic, muffled, echo, distorted'
video_frame_height_width = [512, 992]
seed = 103
slg_layer = 11
audio_guidance_scale = 3.0
video_guidance_scale = 4.0
output_dir = './outputs'

##
video_h, video_w = video_frame_height_width
snap_area = max(video_h * video_w, 720 * 720)
video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area=snap_area)
video_latent_h, video_latent_w = video_h // 16, video_w // 16

scheduler_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
scheduler_video.set_timesteps(sample_steps, device=device, shift=shift)
timesteps_video = scheduler_video.timesteps
scheduler_audio = FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
scheduler_audio.set_timesteps(sample_steps, device=device, shift=shift)
timesteps_audio = scheduler_audio.timesteps



text_embeddings = text_model([text_prompt, video_negative_prompt, audio_negative_prompt], text_model.device)
text_embeddings = [emb.to(target_dtype).to(device) for emb in text_embeddings]
text_embeddings_audio_pos = text_embeddings[0]
text_embeddings_video_pos = text_embeddings[0]
text_embeddings_video_neg = text_embeddings[1]
text_embeddings_audio_neg = text_embeddings[2]


video_noise = torch.randn((video_latent_channel, video_latent_length, video_latent_h, video_latent_w), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed))
audio_noise = torch.randn((audio_latent_length, audio_latent_channel), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed))
        
max_seq_len_audio = audio_noise.shape[0]
_patch_size_h, _patch_size_w = model.video_model.patch_size[1], model.video_model.patch_size[2]
max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h * _patch_size_w)


with torch.amp.autocast('cuda', enabled=target_dtype != torch.float32, dtype=target_dtype):
    for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio)), 
                            total=len(timesteps_video), desc="采样进度"):
        # print(video_noise.mean(), video_noise[0], audio_noise.mean(), audio_noise[0])
        timestep_input = torch.full((1,), t_v, device=device)
                
        # 正向传播
        pos_forward_args = {
            'audio_context': [text_embeddings_audio_pos],
            'vid_context': [text_embeddings_video_pos],
            'vid_seq_len': max_seq_len_video,
            'audio_seq_len': max_seq_len_audio,
            'first_frame_is_clean': False, # is_i2v == False
        }
        pred_vid_pos, pred_audio_pos = model(
            vid=[video_noise],
            audio=[audio_noise],
            t=timestep_input,
            **pos_forward_args
        )
                

        neg_forward_args = {
            'audio_context': [text_embeddings_audio_neg],
            'vid_context': [text_embeddings_video_neg],
            'vid_seq_len': max_seq_len_video,
            'audio_seq_len': max_seq_len_audio,
            'first_frame_is_clean': False, # is_i2v == False,
            'slg_layer': slg_layer
        }  
        pred_vid_neg, pred_audio_neg = model(
            vid=[video_noise],
            audio=[audio_noise],
            t=timestep_input,
            **neg_forward_args
        )
                
        # 应用分类器自由引导
        pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
        pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                
        video_noise = scheduler_video.step(pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
        audio_noise = scheduler_audio.step(pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)

# 解码音频
audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
generated_audio = vae_model_audio.wrapped_decode(audio_latents_for_vae)
generated_audio = generated_audio.squeeze().cpu().float().numpy()
        
# 解码视频
video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
generated_video = vae_model_video.wrapped_decode(video_latents_for_vae)
generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w




output_path = 'test.mp4'
save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)