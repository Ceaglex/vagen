
import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import imageio
import torchvision.transforms as transforms
import random


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
                 target_duration=5.0):
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
        return video


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        try:
            video_path, info = self.samples[idx]

            # with open("debug_read.txt", 'a') as file:
            #     file.write(f"{video_path}\n")

            if self.load_mode == 'raw_video':
                return self.load_raw_video(video_path, info)
            elif self.load_mode == 'video_latent':
                return self.load_video_latent(video_path, info)
            elif self.load_mode == 'audio_latent':
                return self.load_audio_latent(video_path, info)
            elif self.load_mode == 'audio_waveform':
                return self.load_audio_waveform(video_path, info)
            elif self.load_mode == 'video_audio':
                return self.load_raw_video_audio(video_path, info)
            else:
                raise NotImplementedError(f"Load mode {self.load_mode} not implemented.")
            

        except BaseException as e:  
            # print(f"Fail to load file {self.samples[idx]}")
            index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(index)
        


    def load_raw_video(self, video_path, info):
        prompt = info['video_caption']
        video = self.read_video(video_path)  # [T, C, H, W]
        if random.random() < self.uncond_prob:
            prompt = ""
            
        return {
            "video_path": video_path,
            "video_frames": video,
            "prompt": prompt,
            # "negetive_prompt": self.negetive_prompt,
            # "negetive_prompt_embed": self.negetive_prompt_embed if self.negetive_prompt_embed is not None else 0
        }
    
    def load_video_latent(self, video_path, info):
        prompt = info['video_caption']
        video_latent = torch.load(info['video_latent'])  # [C, T, H, W]
        if random.uniform(0,1) < self.uncond_prob:
            prompt = ""

        return {
            "video_path": video_path,
            "video_latent": video_latent,
            "prompt": prompt,
            # "negetive_prompt": self.negetive_prompt,
            # "negetive_prompt_embed": self.negetive_prompt_embed if self.negetive_prompt_embed is not None else 0
        }
    
    def load_audio_latent(self, video_path, info):
        prompt = info['audio_caption']
        audio_latent = torch.load(info['audio_latent'])  # [C, T, H, W]
        if random.uniform(0,1) < self.uncond_prob:
            prompt = ""

        return {
            "video_path": video_path,
            "audio_latent": audio_latent,
            "prompt": prompt,
            # "negetive_prompt": self.negetive_prompt,
            # "negetive_prompt_embed": self.negetive_prompt_embed if self.negetive_prompt_embed is not None else 0
        }


    def load_audio_waveform(self, video_path, info):
        prompt = info['audio_caption']
        # prompt = info['label'] ####

        waveform, sr = torchaudio.load(video_path)
        
        # Convert to target number of channels (mono or stereo)
        num_channels = waveform.shape[0]
        if num_channels != self.target_channels:
            if self.target_channels == 1 and num_channels > 1:
                waveform = waveform[0:1]
            elif self.target_channels == 2 and num_channels == 1:
                waveform = waveform.repeat(2, 1)
            else:
                raise
    
        # Calculate target number of samples
        target_samples = int(self.target_duration * sr)
        current_samples = waveform.shape[1]
        if current_samples > target_samples:
            max_start = current_samples - target_samples
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start+target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        # Resample audio to target sample rate
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        if random.uniform(0,1) < self.uncond_prob:
            prompt = ""


        return {
            "video_path": video_path,
            "audio_waveform": waveform,
            "prompt": prompt,
            "negetive_prompt": self.negetive_prompt,
        }

    def load_raw_video_audio(self, video_path, info):
        v_prompt = info['video_caption']
        a_prompt = info['audio_caption']


        # Audio Waveform
        waveform, sr = torchaudio.load(video_path)
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

        target_samples = int(self.target_duration * sr)
        current_samples = waveform.shape[1]
        if current_samples > target_samples:
            max_start = current_samples - target_samples
            start = random.randint(0, max_start-1)
            end = start+target_samples
            waveform = waveform[:, start:end]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            start = 0
            end = current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        # Video Pixel
        start_duration = start / sr
        video_pixel = self.read_video(video_path, start_duration = start_duration)  # [T, C, H, W]


        # TODO: Better Uncond Configuration
        if random.uniform(0,1) < self.uncond_prob:
            v_prompt = ""
        if random.uniform(0,1) < self.uncond_prob:
            a_prompt = ""


        return {
            "video_path": video_path,
            "video_pixel": video_pixel,
            "audio_waveform": waveform,
            "v_prompt": v_prompt,
            "a_prompt": a_prompt,
        }



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
        # negetive_prompt=args.negetive_prompt if 'negetive_prompt' in args else '',
        # negetive_prompt_embed=args.negetive_prompt_embed if 'negetive_prompt_embed' in args else None,

    )
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            prefetch_factor=args.prefetch_factor,
                            shuffle=args.shuffle,
                            pin_memory=True,
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