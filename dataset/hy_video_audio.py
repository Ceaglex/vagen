
import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import imageio
import torchvision.transforms as transforms
import random
import copy

class VideoAudioDataset(Dataset):
    def __init__(self, 
                 json_path, 
                 load_mode, 
                 uncond_prob = 0.0,
                 video_size=(832, 480), 
                 fps=16, 
                 duration=5, 
                 max_frames=81, 
                 value_range=(-1, 1), 
                 target_sr=44100,
                 target_channels=2,
                 target_duration=5.0,
                 weights=[10,10,10]):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # loading info
        self.samples = list(self.data.items())
        print(f"Dataset total items {len(self.samples)}")
        self.load_mode = load_mode
        self.uncond_prob = uncond_prob

        # video processing params
        self.video_size = video_size
        self.fps = fps
        self.duration = duration
        self.max_frames = max_frames
        self.value_range = value_range
        self.resize_transform = transforms.Resize((self.video_size[1], self.video_size[0]), antialias=True)

        # audio waveform
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.target_duration = target_duration

        # model
        self.modes = ["mask_video", "mask_audio", "shift", "replace"]
        self.weights = weights



    def read_audio(self, audio_path, start_duration = None):
        waveform, sr = torchaudio.load(audio_path, backend='ffmpeg')
        num_channels = waveform.shape[0]
        if num_channels != self.target_channels:
            if self.target_channels == 1 and num_channels > 1:
                waveform = waveform[0:1]
            elif self.target_channels == 2 and num_channels == 1:
                waveform = waveform.repeat(2, 1)
            else:
                raise
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr


        current_samples = waveform.shape[1]
        target_samples = int(self.target_duration * sr)
        if start_duration is not None:
            start = int(start_duration * sr)
            end = min(start+target_samples, current_samples)
            waveform = waveform[:, start:end]        
        elif current_samples > target_samples:
            max_start = current_samples - target_samples
            start = random.randint(0, max_start-1)
            end = start+target_samples
            waveform = waveform[:, start:end]
        elif current_samples <= target_samples:
            padding = target_samples - current_samples
            start = 0
            end = current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        start_duration = start / sr
        return waveform, start_duration



    def read_video(self, video_path, start_duration = 0.0):

        reader = imageio.get_reader(video_path, fps=self.fps)
        frames = [torch.tensor(frame).to(torch.float32) for frame in reader]
        reader.close()
        video = torch.stack(frames)  # Shape: [T, H, W, C]
        video = video / 255.0
        video = video * (self.value_range[1] - self.value_range[0]) + self.value_range[0]
        
        if self.video_size is not None:
            video = video.permute(0, 3, 1, 2)
            video = torch.stack([self.resize_transform(frame) for frame in video])
            video = video.permute(0, 2, 3, 1)
        
        # Ensure number of frames is 4n+1 by trimming excess frames
        start_frames = int(start_duration * self.fps)
        num_frames = video.shape[0] - start_frames
        target_frames = num_frames - (num_frames - 1) % 4  # Closest 4n+1
        target_frames = min(target_frames, self.max_frames)

        video = video[start_frames : start_frames + target_frames]    
        if target_frames < self.max_frames:
            print(f"Pad Frame from {video.shape[0]} to {num_frames}")
            video = torch.cat([video, video[-1:].repeat(self.max_frames - target_frames, 1, 1, 1)], dim=0)



        # Permute to [C, T, H, W]
        video = video.permute(3, 0, 1, 2)
        return video, start_duration


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        try:
            video_path, info = self.samples[idx]

            # with open("debug_read.txt", 'a') as file:
            #     file.write(f"{video_path}\n")

            if self.load_mode == 'video_audio':
                return self.load_raw_video_audio(video_path, info)
            elif self.load_mode == 'video_audio_dpo':
                return self.load_raw_video_audio_shift(video_path, info)
            else:
                raise NotImplementedError(f"Load mode {self.load_mode} not implemented.")
            

        except BaseException as e:  
            # print(f"Fail to load file {self.samples[idx]}")
            index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(index)
        


    def load_raw_video_audio(self, video_path, info):
        v_prompt = info['video_caption']
        a_prompt = info['audio_caption']
        waveform, raw_sr = torchaudio.load(video_path, backend='ffmpeg')
        max_start = waveform.shape[1] - int(self.target_duration * raw_sr)
        start_duration = random.randint(0, max_start-1) / raw_sr
        
        waveform1, start_duration = self.read_audio(video_path, start_duration)
        video_pixel1, start_duration = self.read_video(video_path, start_duration = start_duration)


        if random.uniform(0,1) < self.uncond_prob:
            v_prompt = ""
        if random.uniform(0,1) < self.uncond_prob:
            a_prompt = ""

        return {
            "video_path": video_path,
            "video_pixel": video_pixel1,
            "audio_waveform": waveform1,
            "v_prompt": v_prompt,
            "a_prompt": a_prompt,
        }


    # def contruct_lose_vapair(self)


    def load_raw_video_audio_shift(self, video_path, info):
        v_prompt = info['video_caption']
        a_prompt = info['audio_caption']
        waveform, raw_sr = torchaudio.load(video_path, backend='ffmpeg')
        max_start = waveform.shape[1] - int(self.target_duration * raw_sr)
        start_duration = random.randint(0, max_start-1) / raw_sr
        # mode = random.choice(self.modes) #
        mode = random.choices(self.modes, weights=self.weights, k=1)[0] 


        ## Shift
        if mode == 'shift':
            start_duration, start_duration2 = self.sample_two_starts_discrete(max_start, raw_sr, min_gap_seconds = 1)
            waveform1, start_duration = self.read_audio(video_path, start_duration)
            video_pixel1, start_duration = self.read_video(video_path, start_duration = start_duration)
            waveform2 = copy.deepcopy(waveform1)
            video_pixel2 = copy.deepcopy(video_pixel1)
            if random.uniform(0,1) < 0.5:
                waveform2, start_duration2 = self.read_audio(video_path, start_duration2)  
            else:
                video_pixel2, start_duration2 = self.read_video(video_path, start_duration = start_duration2)  
                
        ## Mask Video
        elif mode == 'mask_video':
            waveform1, start_duration = self.read_audio(video_path, start_duration)
            video_pixel1, start_duration = self.read_video(video_path, start_duration = start_duration)
            waveform2 = copy.deepcopy(waveform1)
            video_pixel2 = copy.deepcopy(video_pixel1)

            T = video_pixel2.shape[1]
            start_frame = random.randint(0, max(0, T // 3))
            end_frame = random.randint((T * 2) // 3, T - 1)
            
            # # Get start and end frames
            # start_frame_data = video_pixel2[:, start_frame:start_frame+1, :, :]  # [C, 1, H, W]
            # end_frame_data = video_pixel2[:, end_frame:end_frame+1, :, :]  # [C, 1, H, W]
            # alpha = torch.linspace(0, 1, end_frame - start_frame + 1).view(1, -1, 1, 1).to(video_pixel2.device)  # [1, num_interp_frames, 1, 1]
            # interp_frames = (1 - alpha) * start_frame_data + alpha * end_frame_data  # [C, num_interp_frames, H, W]
            # video_pixel2[:, start_frame:end_frame+1, :, :] = interp_frames


            num_source_frames = random.randint(1, min(6, T // 10))
            source_frame_indices = torch.linspace(0, num_source_frames-1, num_source_frames).long()
            source_frames = video_pixel2[:, source_frame_indices, :, :]  # [C, num_source_frames, H, W]
            reversed_frames = torch.flip(source_frames, dims=[1])  # Reverse along time dimension
            pattern_frames = torch.cat([source_frames, reversed_frames], dim=1)  # [C, 2*num_source_frames, H, W]
            num_patterns = (T + pattern_frames.shape[1] - 1) // pattern_frames.shape[1]  # Ceiling division
            video_pixel2 = pattern_frames.repeat(1, num_patterns, 1, 1)[:, :T, :, :]  # [C, T, H, W]

            # repeat_per_frame = T // num_source_frames
            # remainder = T % num_source_frames
            # slow_motion_frames = []
            # for i in range(num_source_frames):
            #     frames_to_repeat = repeat_per_frame + (1 if i < remainder else 0)
            #     slow_motion_frames.append(source_frames[:, i:i+1, :, :].repeat(1, frames_to_repeat, 1, 1))
            # video_pixel2 = torch.cat(slow_motion_frames, dim=1)  # [C, T, H, W]

        ## Mask Audio
        elif mode == 'mask_audio':
            waveform1, start_duration = self.read_audio(video_path, start_duration)
            video_pixel1, start_duration = self.read_video(video_path, start_duration = start_duration)
            waveform2 = copy.deepcopy(waveform1)
            video_pixel2 = copy.deepcopy(video_pixel1)
            waveform2 = waveform2 * 0.002
            
            # S = waveform2.shape[1]
            # mask_start_sample = random.randint(0, S // 3)
            # mask_end_sample = random.randint(2 * S // 3, S)
            # length = mask_end_sample - mask_start_sample
            # t = torch.linspace(0, 1, steps=length, device=waveform2.device, dtype=waveform2.dtype)
            # min_amp = 0.1 
            # env = 1.0 - (1.0 - min_amp) * 0.5 * (1.0 - torch.cos(2 * torch.pi * t))
            # waveform2[:, mask_start_sample:mask_end_sample] = waveform2[:, mask_start_sample:mask_end_sample] * 0.001

        ## Replace
        elif mode == 'replace':
            video_pixel1, start_duration = self.read_video(video_path, start_duration = start_duration)
            waveform1, start_duration = self.read_audio(video_path, start_duration)
            waveform2 = copy.deepcopy(waveform1)
            video_pixel2 = copy.deepcopy(video_pixel1)

            idx_ = random.randint(0, len(self.samples)) 
            replace_video_path, info = self.samples[idx_]
            waveform_new, raw_sr = torchaudio.load(replace_video_path, backend='ffmpeg')
            max_start = waveform_new.shape[1] - int(self.target_duration * raw_sr)
            start_duration = random.randint(0, max_start-1) / raw_sr
            if random.uniform(0,1) < 0.5:
                waveform2, start_duration2 = self.read_audio(replace_video_path, start_duration)  
            else:
                video_pixel2, start_duration2 = self.read_video(replace_video_path, start_duration = start_duration)  

        ## 
        else:
            raise NotImplementedError("Other Lose Pair Contruction Mode is not Supported.")



        if random.uniform(0,1) < self.uncond_prob:
            v_prompt = ""
        if random.uniform(0,1) < self.uncond_prob:
            a_prompt = ""


        return {
            "mode": mode,
            "video_path": video_path,
            "video_pixel": video_pixel1,
            "video_pixel_lose": video_pixel2,
            "audio_waveform": waveform1,
            "audio_waveform_lose": waveform2,
            # "start_duration1": start_duration1,
            # "start_duration2": start_duration2, 
            "v_prompt": v_prompt,
            "a_prompt": a_prompt,
        }



    def sample_two_starts_discrete(self, max_start, raw_sr, min_gap_seconds=1.0):
        min_interval_samples = raw_sr * min_gap_seconds
        if max_start <= min_interval_samples:
            pos1 = random.randint(0, max_start-1) 
            pos2 = random.randint(0, max_start-1) 
        else:
            pos1 = random.randint(0, max_start - min_interval_samples - 1)
            pos2 = random.randint(pos1 + min_interval_samples, max_start - 1)

        start_duration1 = pos1 / raw_sr
        start_duration2 = pos2 / raw_sr
        return start_duration1, start_duration2



def build_video_loader(args):

    dataset = VideoAudioDataset(
        json_path=args.video_index_file,
        load_mode=args.load_mode,
        uncond_prob=args.uncond_prob if 'uncond_prob' in args else 0.1,
        video_size=args.video_size if 'video_size' in args else (832, 480),
        fps=args.fps if 'fps' in args else 16,
        duration=args.duration if 'duration' in args else 5,
        max_frames=args.max_frames if 'max_frames' in args else 81,
        value_range=args.value_range if 'value_range' in args else (-1, 1),
        target_sr=args.target_sr if 'target_sr' in args else 44100,
        target_channels=args.target_channels if 'target_channels' in args else 2,
        target_duration=args.target_duration if 'target_duration' in args else 5.0,
        weights=args.weights if 'weights' in args else [10,10,10]
        # negetive_prompt=args.negetive_prompt if 'negetive_prompt' in args else '',
        # negetive_prompt_embed=args.negetive_prompt_embed if 'negetive_prompt_embed' in args else None,

    )
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            prefetch_factor=args.prefetch_factor,
                            shuffle=args.shuffle,
                            pin_memory=False,
                            )
    return dataloader



if __name__ == "__main__":

    from omegaconf import OmegaConf
    config = OmegaConf.load("/home/chengxin/chengxin/veo3/config/ttv_tuning.yaml")
    dataloader = build_video_loader(config.hy_dataloader)
    for i, batch in enumerate(dataloader):
        print(i, batch['video_latent'].shape, batch['prompt'])
        if i > 10:
            break