"""Isolated Mizar single-audio CED mapper plus the unchanged full SmolLM2."""
import torch
from torch import nn

from models.audio import get_audio_encoder
from models.decoder import get_decoder


class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name, d_in, d_out, use_pretrained_audioencoder,
                 freeze_audio_encoder_weights, pretrained_audioencoder_path):
        super().__init__()
        if not use_pretrained_audioencoder or not freeze_audio_encoder_weights:
            raise ValueError("this route requires a pretrained, frozen CED-Small encoder")
        audio_encoder, _ = get_audio_encoder(audioenc_name)
        self.base = audio_encoder(
            encoder_config={"pretrained_audioencoder_path": pretrained_audioencoder_path}, d_out=d_out)
        # CEDTemporalMapper already produces 576-dimensional LM inputs.
        self.projection = nn.Identity()
        self.single_audio_mode = True

    def forward(self, waveform):
        output = self.base(waveform)
        return self.projection(output["embedding"]), output["clipwise_output"], output


class Mellow(nn.Module):
    def __init__(self, audioenc_name, d_in, text_decoder, prefix_length,
                 freeze_text_decoder_weights, d_out, use_pretrained_audioencoder,
                 freeze_audio_encoder_weights, pretrained_audioencoder_path=None):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            audioenc_name, d_in, d_out, use_pretrained_audioencoder,
            freeze_audio_encoder_weights, pretrained_audioencoder_path)
        self.caption_decoder = get_decoder("Decoder")(
            text_decoder, prefix_length, freeze_text_decoder_weights)

    def forward(self, input_dict):
        # Rejoin the original fixed20 waveform BEFORE CED resampling/mel/BN.
        audio_20s = torch.cat((input_dict["audio1"], input_dict["audio2"]), dim=-1)
        audio_embed, _, output = self.audio_encoder(audio_20s)
        return self.caption_decoder(audio_embed, None, input_dict["input"], input_dict["answer"],
                                    audio_grids=output["layer_grids"])

    def generate_prefix_inference(self, input_dict):
        audio_20s = torch.cat((input_dict["audio1"], input_dict["audio2"]), dim=-1)
        audio_embed, _, output = self.audio_encoder(audio_20s)
        prefix = self.caption_decoder.generate_prefix_inference(audio_embed, None, input_dict["input"])
        # Full-prefix inference must pass this context back to the LM on every step.
        context = {"audio_grids": output["layer_grids"],
                   "prompt_mask": input_dict["input"]["attention_mask"]}
        return prefix, output, context
