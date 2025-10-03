# import pdb; pdb.set_trace()
import os
import json
import torch
import torchaudio
from diffusers.utils import export_to_video
from worker.base import prepare_config, prepare_everything
from model.jointva.pipeline_jointva import JointVAPipeline





def main(args, accelerator):

    load_dtype = torch.float32  
    infer_dtype = torch.bfloat16

    pipeline = JointVAPipeline(
        load_dtype = load_dtype,
        infer_dtype = infer_dtype,
        device = accelerator.device,
        bridge_config = getattr(args, 'bridge_config', None),
        bridge_weights_path = getattr(args, 'bridge_safetensors_path', None),
        audio_vae_path = getattr(args, 'audio_pretrained_model_name_or_path', None),
        audio_transformer_path = getattr(args, 'audio_pretrained_model_name_or_path', None),
        audio_transformer_weights_path = getattr(args, 'audio_transformer_safetensors_path', None),
        audio_projection_model_path = getattr(args, 'audio_pretrained_model_name_or_path', None),
        audio_text_encoder_path = getattr(args, 'audio_pretrained_model_name_or_path', None),
        audio_tokenizer_path = getattr(args, 'audio_pretrained_model_name_or_path', None),
        video_vae_path = getattr(args, 'video_pretrained_model_name_or_path', None),
        video_transformer_path = getattr(args, 'video_pretrained_model_name_or_path', None),
        video_transformer_weights_path = getattr(args, 'video_transformer_safetensors_path', None),
        video_text_encoder_path = getattr(args, 'video_pretrained_model_name_or_path', None),
        video_tokenizer_path = getattr(args, 'video_pretrained_model_name_or_path', None),
        scheduler_config = getattr(args, 'scheduler', None),
    )



    if args.infer_mode == 'tta':
        with open(args.tta_data_info.prompt_index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = list(data.items())
        idx = accelerator.process_index if accelerator.process_index < len(data) else 0
        step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
        data = data[idx::step_range]
        output_dir = args.tta_data_info.output_dir
        os.makedirs(output_dir, exist_ok=True)
        generator = torch.Generator("cuda").manual_seed(args.tta_infer_config.seed)

        for path, info in data:
            prompts = [info['audio_caption']]
            gen_audio = pipeline.text_to_audio(
                a_prompts=prompts,
                generator=generator,
                **args.tta_infer_config,
            )
            gen_audio = gen_audio[0] # bs = 1
            for i in range(len(gen_audio)): # num_audio = 1
                torchaudio.save(f"{output_dir}/{path.split('/')[-1][:-4]}.wav", gen_audio[i].cpu().to(torch.float32), sample_rate= args.tta_data_info.sr)



    elif args.infer_mode == 'ttv':
        with open(args.ttv_data_info.prompt_index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = list(data.items())
        idx = accelerator.process_index if accelerator.process_index < len(data) else 0
        step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
        data = data[idx::step_range]
        output_dir = args.ttv_data_info.output_dir
        os.makedirs(output_dir, exist_ok=True)
        generator = torch.Generator("cuda").manual_seed(args.ttv_infer_config.seed)

        for path, info in data:
            prompts = [info['video_caption']]
            gen_video = pipeline.text_to_video(
                v_prompts=prompts,
                v_generator=generator,
                **args.ttv_infer_config,
            )
            gen_video = gen_video[0] # bs = 1
            for i in range(len(gen_video)): # num_video = 1
                export_to_video(gen_video[i], f"{output_dir}/{path.split('/')[-1][:-4]}.mp4", fps=args.ttv_data_info.fps)



    elif args.infer_mode == 'ttva':
        with open(args.data_info.prompt_index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = list(data.items())
        idx = accelerator.process_index if accelerator.process_index < len(data) else 0
        step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
        data = data[idx::step_range]
        output_dir = args.data_info.output_dir
        os.makedirs(output_dir, exist_ok=True)
        v_generator = torch.Generator(accelerator.device).manual_seed(args.ttav_infer_config.seed)
        a_generator = torch.Generator(accelerator.device).manual_seed(args.ttav_infer_config.seed)

        for path, info in data:
            va_prompts = [[info['video_caption'], info['audio_caption']]]
            gen_video_audio = pipeline.text_to_video_audio(
                va_prompts=va_prompts,
                v_generator=v_generator,
                a_generator=a_generator,
                **args.ttav_infer_config,
            )
            gen_video, gen_audio = gen_video_audio[0] # bs = 1
            for i in range(len(gen_video)): # num_video = 1
                export_to_video(gen_video[i], f"{output_dir}/{path.split('/')[-1][:-4]}.mp4", fps=args.data_info.video_info.fps)
                torchaudio.save(f"{output_dir}/{path.split('/')[-1][:-4]}.wav", gen_audio[i].cpu().to(torch.float32), sample_rate= args.data_info.audio_info.sr)
            # break #####


    else:
        raise NotImplementedError()





if __name__ == "__main__":
    args = prepare_config()
    args, accelerator = prepare_everything(args)
    main(args, accelerator)