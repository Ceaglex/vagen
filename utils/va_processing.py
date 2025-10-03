import torch
import torch.fft as fft
from librosa.filters import mel as librosa_mel
from transformers import SpeechT5HifiGan
from diffusers.models import AutoencoderKL
import torchaudio
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip



def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        print(video.duration, audio.duration)

        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)  # 裁剪音频到视频时长
        video_with_audio = video.with_audio(audio)
        video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")       
        video.close()
        audio.close() 
        video_with_audio.close()
        
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
