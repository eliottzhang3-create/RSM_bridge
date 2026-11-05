from models.ced import CEDSmallWrapper

def get_audio_encoder(name: str):
    if name == "CEDSmall":
        return CEDSmallWrapper, 384
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))
