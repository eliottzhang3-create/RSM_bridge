from models.usad2 import USAD2Wrapper


def get_audio_encoder(name: str):
    if name == "USAD2-Small":
        return USAD2Wrapper, 384
    raise ValueError(f"unsupported audio encoder: {name}")
