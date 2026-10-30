"""Shared, twice-applied CED depth selection and text/audio cross attention.

CED grids are frozen block outputs BEFORE the encoder's final LayerNorm.
Only the Q input receives a round bias. Gates and Q read normalized text,
while the residual base stays unchanged. Prompt pooling excludes answer/PAD.
"""
import torch
from torch import nn


class MultilayerAudioCrossAttention(nn.Module):
    def __init__(self, width=576, num_heads=9, audio_tokens=126, prompt_tokens=256):
        super().__init__()
        if width % num_heads:
            raise ValueError("cross attention width must be divisible by num_heads")
        self.width = width
        self.audio_tokens = audio_tokens
        self.prompt_tokens = prompt_tokens
        self.text_start = audio_tokens + 1
        self.residual_scale = 0.3
        # All new modules initialize on CPU without advancing the caller's RNG.
        # Registration happens after strict loading of every original LM tensor.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleList([
                nn.Linear(1536, width, device="cpu", dtype=torch.float32) for _ in range(4)])
            self.gate1 = nn.Linear(width, 4, device="cpu", dtype=torch.float32)
            self.gate2 = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
            self.cross_attention = nn.MultiheadAttention(
                width, num_heads, dropout=0., bias=True, batch_first=True,
                device="cpu", dtype=torch.float32)
            # Store each bias as a 1-D parameter: the inherited no-decay rule applies.
            self.round_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(width, dtype=torch.float32)) for _ in range(2)])
            for projection in self.projections:
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            nn.init.zeros_(self.gate1.weight)
            nn.init.zeros_(self.gate1.bias)
            nn.init.zeros_(self.gate2.weight)
            nn.init.zeros_(self.gate2.bias)
            nn.init.zeros_(self.cross_attention.in_proj_bias)
            nn.init.zeros_(self.cross_attention.out_proj.bias)
            # Constant LN initialization consumes no RNG; preserve every existing
            # projection/attention initialization and the caller's RNG exactly.
            self.audio_input_norms = nn.ModuleList([
                nn.LayerNorm(1536, eps=1e-5, device="cpu", dtype=torch.float32) for _ in range(4)])
            self.audio_output_norms = nn.ModuleList([
                nn.LayerNorm(width, eps=1e-5, device="cpu", dtype=torch.float32) for _ in range(4)])
            self.text_norm = nn.LayerNorm(width, eps=1e-5, device="cpu", dtype=torch.float32)
            self.output_norm = nn.LayerNorm(
                width, eps=1e-5, elementwise_affine=False, device="cpu", dtype=torch.float32)
        self.collect_diagnostics = False
        self.last_diagnostics = []

    def project_audio(self, grids, hidden_states):
        expected = (hidden_states.shape[0], 4, self.audio_tokens, 1536)
        if grids is None or tuple(grids.shape) != expected:
            raise ValueError(f"CED four-depth grids must have shape {expected}")
        if grids.device != hidden_states.device or grids.dtype != hidden_states.dtype:
            raise ValueError("CED grids and LM inputs must share FP32 dtype/device")
        self.last_diagnostics = []
        return torch.stack([self.audio_output_norms[index](
                                projection(self.audio_input_norms[index](grids[:, index])))
                            for index, projection in enumerate(self.projections)], dim=1)

    def pool_prompt(self, text_states, prompt_mask):
        """Pool only valid prompt tokens from already normalized text states."""
        if text_states.ndim != 3 or text_states.shape[-1] != self.width:
            raise ValueError("invalid cross attention hidden shape")
        if text_states.shape[1] < self.prompt_tokens:
            raise ValueError("cross attention requires the complete fixed256 prompt")
        if prompt_mask is None or tuple(prompt_mask.shape) != (text_states.shape[0], self.prompt_tokens):
            raise ValueError("Gate1 requires the tokenizer's fixed256 prompt attention_mask")
        mask = prompt_mask.to(device=text_states.device, dtype=text_states.dtype)
        if not bool(((mask == 0) | (mask == 1)).all().item()):
            raise ValueError("prompt attention_mask must be binary")
        counts = mask.sum(dim=1, keepdim=True)
        if not bool((counts > 0).all().item()):
            raise ValueError("Gate1 cannot pool an empty prompt")
        prompt = text_states[:, :self.prompt_tokens]
        return (prompt * mask.unsqueeze(-1)).sum(dim=1) / counts

    def forward(self, hidden_states, audio_bank, prompt_mask, round_index):
        if round_index not in (0, 1):
            raise ValueError("only the two middle-loop entrances receive cross attention")
        text = hidden_states[:, self.text_start:]
        normalized_text = self.text_norm(text)
        summary = self.pool_prompt(normalized_text, prompt_mask)
        layer_weights = torch.softmax(self.gate1(summary), dim=-1)
        audio = (audio_bank * layer_weights[:, :, None, None]).sum(dim=1)
        # Q-only bias: never add to Gate1, Gate2, K, V or the residual base.
        query = normalized_text + self.round_bias[round_index]
        delta, _ = self.cross_attention(query, audio, audio, need_weights=False)
        normalized_delta = self.output_norm(delta)
        gate = torch.tanh(self.gate2(normalized_text))
        residual = self.residual_scale * gate * normalized_delta
        updated_text = text + residual
        if self.collect_diagnostics:
            # Only detached small scalars survive a forward; no graph or audio bank.
            stats = torch.cat((layer_weights.detach().mean(dim=0),
                               torch.stack((gate.detach().mean(), gate.detach().abs().mean(),
                                            delta.detach().square().mean().sqrt(),
                                            residual.detach().square().mean().sqrt(),
                                            normalized_delta.detach().square().mean().sqrt(),
                                            residual.detach().square().mean(dim=-1).sqrt().max()))))
            self.last_diagnostics.append((round_index, stats))
        return torch.cat((hidden_states[:, :self.text_start], updated_text), dim=1)

    def diagnostics(self):
        return [{"round": index + 1, "mean_depth_weights": values[:4].cpu().tolist(),
                 "gate_mean": float(values[4]), "gate_abs_mean": float(values[5]),
                 "cross_attention_rms": float(values[6]), "residual_rms": float(values[7]),
                 "normalized_cross_attention_rms": float(values[8]),
                 "residual_token_rms_max": float(values[9]), "residual_scale": self.residual_scale}
                for index, values in self.last_diagnostics]
