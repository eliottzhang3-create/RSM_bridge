"""S3: SoundFile -> resample32k -> mono -> CENTER20 -> right-pad."""
import torch
import torchaudio
from utils.audio_io import load_audio
SAMPLE_RATE,HALF_SAMPLES=32000,320000

def fixed_len(waveform,target_len):
    if waveform.numel()>target_len:
        start=(waveform.numel()-target_len)//2
        waveform=waveform[start:start+target_len]
    return torch.nn.functional.pad(waveform,(0,max(0,target_len-waveform.numel()))).contiguous()

def load_mono(path):
    waveform,rate=load_audio(path,channels_first=True)
    if rate!=SAMPLE_RATE:waveform=torchaudio.transforms.Resample(rate,SAMPLE_RATE)(waveform)
    if waveform.ndim!=2 or waveform.shape[0]<1:raise ValueError(f"invalid audio shape: {path}")
    waveform=waveform.mean(dim=0).contiguous()
    if not waveform.numel() or not torch.isfinite(waveform).all():raise ValueError(f"empty/nonfinite audio: {path}")
    return waveform

def load_fixed20(path1,path2=""):
    if path2:raise ValueError("S3 requires one original recording, not chunk-pair input")
    whole=fixed_len(load_mono(path1),2*HALF_SAMPLES)
    return whole[:HALF_SAMPLES].contiguous(),whole[HALF_SAMPLES:].contiguous()
