import torch
import torch.fft as fft
from librosa.filters import mel as librosa_mel
from transformers import SpeechT5HifiGan
from diffusers.models import AutoencoderKL
import torchaudio
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip
import re
import cv2
from PIL import Image

import tempfile
from typing import Optional
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip
from scipy.io import wavfile
import math
import random




def preprocess_image_tensor(image_path, device, target_dtype, h_w_multiple_of=32, resize_total_area=720*720):
    """Preprocess video data into standardized tensor format and (optionally) resize area."""
    def _parse_area(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]) * int(val[1])
        if isinstance(val, str):
            m = re.match(r"\s*(\d+)\s*[x\*\s]\s*(\d+)\s*$", val, flags=re.IGNORECASE)
            if m:
                return int(m.group(1)) * int(m.group(2))
            if val.strip().isdigit():
                return int(val.strip())
        raise ValueError(f"resize_total_area={val!r} is not a valid area or WxH.")

    def _best_hw_for_area(h, w, area_target, multiple):
        if area_target <= 0:
            return h, w
        ratio_wh = w / float(h)
        area_unit = multiple * multiple
        tgt_units = max(1, area_target // area_unit)
        p0 = max(1, int(round(np.sqrt(tgt_units / max(ratio_wh, 1e-8)))))
        candidates = []
        for dp in range(-3, 4):
            p = max(1, p0 + dp)
            q = max(1, int(round(p * ratio_wh)))
            H = p * multiple
            W = q * multiple
            candidates.append((H, W))
        scale = np.sqrt(area_target / (h * float(w)))
        H_sc = max(multiple, int(round(h * scale / multiple)) * multiple)
        W_sc = max(multiple, int(round(w * scale / multiple)) * multiple)
        candidates.append((H_sc, W_sc))
        def score(HW):
            H, W = HW
            area = H * W
            return (abs(area - area_target), abs((W / max(H, 1e-8)) - ratio_wh))
        H_best, W_best = min(candidates, key=score)
        return H_best, W_best

    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        assert isinstance(image_path, Image.Image)
        if image_path.mode != "RGB":
            image_path = image_path.convert("RGB")
        image = np.array(image_path)

    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image).float().to(device, dtype=target_dtype).unsqueeze(0) ## b c h w
    image_tensor = image_tensor * 2.0 - 1.0 ## -1 to 1

    _, c, h, w = image_tensor.shape
    area_target = _parse_area(resize_total_area)
    if area_target is not None:
        target_h, target_w = _best_hw_for_area(h, w, area_target, h_w_multiple_of)
    else:
        target_h = (h // h_w_multiple_of) * h_w_multiple_of
        target_w = (w // h_w_multiple_of) * h_w_multiple_of

    target_h = max(h_w_multiple_of, int(target_h))
    target_w = max(h_w_multiple_of, int(target_w))

    if (h != target_h) or (w != target_w):
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )

    return image_tensor




def snap_hw_to_multiple_of_32(h: int, w: int, area = 720 * 720) -> tuple[int, int]:
    """
    Scale (h, w) to match a target area if provided, then snap both
    dimensions to the nearest multiple of 32 (min 32).
    
    Args:
        h (int): original height
        w (int): original width
        area (int, optional): target area to scale to. If None, no scaling is applied.
    
    Returns:
        (new_h, new_w): dimensions adjusted
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got {(h, w)}")

    # If a target area is provided, rescale h, w proportionally
    if area is not None and area > 0:
        current_area = h * w
        scale = math.sqrt(area / float(current_area))
        h = int(round(h * scale))
        w = int(round(w * scale))

    # Snap to nearest multiple of 32
    def _n32(x: int) -> int:
        return max(32, int(round(x / 32)) * 32)
    return _n32(h), _n32(w)



def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.

    Args:
        output_path (str): Path to the output MP4 file.
        video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                  Values can be in range [-1, 1] or [0, 255].
        audio_numpy (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
        sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
        fps (int): Frames per second for the video. Defaults to 24.

    Returns:
        str: Path to the saved MP4 file.
    """
    try:
        # Validate inputs
        assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
        assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
        assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

        if audio_numpy is not None:
            assert isinstance(audio_numpy, np.ndarray), "audio_numpy must be a numpy array"
            assert np.abs(audio_numpy).max() <= 1.0, "audio_numpy values must be in range [-1, 1]"

        # Reorder dimensions: (C, F, H, W) → (F, H, W, C)
        video_numpy = video_numpy.transpose(1, 2, 3, 0)

        # Normalize frames if values are in [-1, 1]
        if video_numpy.max() <= 1.0:
            video_numpy = np.clip(video_numpy, -1, 1)
            video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
        else:
            video_numpy = video_numpy.astype(np.uint8)

        # Convert numpy array to a list of frames
        frames = list(video_numpy)
        clip = ImageSequenceClip(frames, fps=fps)

        if audio_numpy is not None:
            audio_path = output_path.replace(".mp4", ".wav")
            wavfile.write(
                audio_path,
                sample_rate,
                (audio_numpy * 32767).astype(np.int16),
            )
            audio_clip = AudioFileClip(audio_path)
            clip.audio = audio_clip

        # if audio_numpy is not None:
        #     with tempfile.NamedTemporaryFile(suffix=f"{random.randint(1, 1000000)}.wav", mode='wb', delete=False) as temp_audio_file:
        #         wavfile.write(
        #             temp_audio_file.name,
        #             sample_rate,
        #             (audio_numpy * 32767).astype(np.int16),
        #         )
        #         audio_clip = AudioFileClip(temp_audio_file.name)
        #         clip.audio = audio_clip

        final_clip = clip
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        final_clip.close()

        return output_path
    except Exception as e:
        print(f"Error when writing into file {output_path} because of {e}")




def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        print(video.duration, audio.duration)

        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)  # 裁剪音频到视频时长
            # audio = audio.subclipped(audio.duration - video.duration, audio.duration)
        # video_with_audio = video.with_audio(audio)
        video.audio = audio
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")       
        video.close()
        audio.close() 
        # video_with_audio.close()
        
        print(f"视频已保存至：{output_path}")
        
    except Exception as e:
        print(f"发生错误：{e}")
        


def extract_batch_mel(waveform, 
                      cut_audio_duration, 
                      sampling_rate, 
                      hop_length,
                      maximum_amplitude,
                      filter_length,
                      n_mel,
                      mel_fmin,
                      mel_fmax,
                      win_length):
    target_mel_length = int(cut_audio_duration * sampling_rate / hop_length)
    waveform = waveform - torch.mean(waveform, dim=1, keepdim=True)
    waveform = waveform / (torch.max(torch.abs(waveform), dim=1).values.unsqueeze(1) + 1e-8)
    waveform = waveform * maximum_amplitude

    waveform = waveform.unsqueeze(0)
    waveform = torch.nn.functional.pad(
        waveform,
        ( int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2), 0, 0),
        mode="reflect",)
    waveform = waveform.squeeze(0)


    mel_basis = librosa_mel(
        sr=sampling_rate,
        n_fft=filter_length,
        n_mels=n_mel,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(waveform.device)
    hann_window = torch.hann_window(win_length).to(waveform.device)


    def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
    stft_spec = torch.stft(
        waveform,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    stft_spec = torch.abs(stft_spec)
    mel = dynamic_range_compression_torch( torch.matmul(mel_basis, stft_spec) )



    def pad_spec(cur_log_mel_spec):
        n_frames = cur_log_mel_spec.shape[-2]
        p = target_mel_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ConstantPad2d((0, 0, 0, p), value=-11)
            cur_log_mel_spec = m(cur_log_mel_spec)
        elif p < 0:
            cur_log_mel_spec = cur_log_mel_spec[..., 0 : target_mel_length, :]

        if cur_log_mel_spec.size(-1) % 2 != 0:
            cur_log_mel_spec = cur_log_mel_spec[..., :-1]
        return cur_log_mel_spec
    log_mel_spec, stft = mel.to(torch.float).transpose(1,2), stft_spec.to(torch.float).transpose(1,2)
    log_mel_spec, stft = pad_spec(log_mel_spec), pad_spec(stft)

    return waveform, log_mel_spec



if __name__ == "__main__":
    audio_vae = AutoencoderKL.from_pretrained(
        # From pretrained
        "/data-04/xihua/data/ckpt/audioldm2/huggingface/vae",
        local_files_only=True,
        scaling_factor=1,
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=False,
        use_safetensors=True,
    )
    vocoder = SpeechT5HifiGan.from_pretrained(
        # From pretrained
        "/data-04/xihua/data/ckpt/audioldm2/huggingface/vocoder",
        local_files_only=True,
        low_cpu_mem_usage=True, 
        ignore_mismatched_sizes=False,
        use_safetensors=True,
    )


    va_path = "/home/chengxin/chengxin/veo3/assets/cache/audio__cES7Twcq18_000006_00.wav"
    save_path = f"test.wav"

    audio_waveform, audio_sr = torchaudio.load(va_path) 
    audio_waveform = audio_waveform[0]  # [channel_num, sample_num] -> [sample_num]
    audio_waveform = torch.stack([audio_waveform])
    audio_waveform, log_mel_spec = extract_batch_mel(audio_waveform, 
                                                     cut_audio_duration = 10, 
                                                     sampling_rate = 16000, 
                                                     hop_length = 160, 
                                                     maximum_amplitude = 0.5,
                                                    filter_length = 1024, 
                                                    n_mel = 64, 
                                                    mel_fmin = 0, 
                                                    mel_fmax = 8000, 
                                                    win_length = 1024)
    log_mel_spec = log_mel_spec.unsqueeze(1)  # [bs, 1, target_mel_length (sr*duration/hop_length, 1000), n_mel (64)]
        
    audio_latent = audio_vae.encode(log_mel_spec.to(audio_vae.encoder.conv_in.weight.dtype)).latent_dist    #  [bs, 8, target_mel_length/4(250), n_mel/4(16)]
    audio_latent = audio_latent.sample()

    mel_spectrogram = audio_vae.decode(audio_latent).sample                             # [bs, 1, target_mel_length(latent_length*4), 64(16*4)]
    gen_audio = vocoder(mel_spectrogram.squeeze(1))                               # [bs, duration*sr+...]

    for i in range(len(gen_audio)):
        torchaudio.save(save_path, gen_audio[i:i+1], sample_rate=16000)
