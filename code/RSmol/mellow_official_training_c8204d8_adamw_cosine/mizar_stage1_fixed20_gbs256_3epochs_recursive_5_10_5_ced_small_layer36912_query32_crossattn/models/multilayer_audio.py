"""Shared, twice-applied CED depth selection and text/audio cross attention.

CED grids are frozen block outputs BEFORE the encoder's final LayerNorm.
Only the Q input receives a round bias. Gates and Q read normalized text,
while the residual base stays unchanged. Prompt pooling excludes answer/PAD.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultilayerAudioCrossAttention(nn.Module):
    def __init__(self, width=576, num_heads=9, audio_tokens=126, prompt_tokens=256, memory_tokens=32):
        super().__init__()
        if width % num_heads:
            raise ValueError("cross attention width must be divisible by num_heads")
        self.width = width
        self.audio_tokens = audio_tokens
        if type(memory_tokens) is not int or memory_tokens < 2:
            raise ValueError("query memory requires at least two learned tokens")
        self.memory_tokens = memory_tokens
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
            # Append new initialization after all v2 modules to preserve their
            # initial tensors as well as the caller's CPU RNG state.
            self.query_tokens = nn.Parameter(torch.empty(memory_tokens, width, dtype=torch.float32))
            nn.init.normal_(self.query_tokens, mean=0., std=0.02)
            self.compression_attention = nn.MultiheadAttention(
                width, num_heads, dropout=0., bias=True, batch_first=True,
                device="cpu", dtype=torch.float32)
            nn.init.zeros_(self.compression_attention.in_proj_bias)
            nn.init.zeros_(self.compression_attention.out_proj.bias)
            self.query_norm = nn.LayerNorm(width, eps=1e-5, device="cpu", dtype=torch.float32)
            self.memory_norm = nn.LayerNorm(
                width, eps=1e-5, elementwise_affine=False, device="cpu", dtype=torch.float32)
        self.collect_diagnostics = False
        self.last_diagnostics = []
        self.last_memory_diagnostics = None

    def project_audio(self, grids, hidden_states):
        expected = (hidden_states.shape[0], 4, self.audio_tokens, 1536)
        if grids is None or tuple(grids.shape) != expected:
            raise ValueError(f"CED four-depth grids must have shape {expected}")
        if grids.device != hidden_states.device or grids.dtype != hidden_states.dtype:
            raise ValueError("CED grids and LM inputs must share FP32 dtype/device")
        self.last_diagnostics = []
        self.last_memory_diagnostics = None
        projected = torch.stack([self.audio_output_norms[index](
                                projection(self.audio_input_norms[index](grids[:, index])))
                            for index, projection in enumerate(self.projections)], dim=1)
        # B-major, depth-minor flattening: every depth reads the SAME original Q
        # independently. Never concatenate depths along K/V or chain Q outputs.
        batch = hidden_states.shape[0]
        source = projected.reshape(batch * 4, self.audio_tokens, self.width)
        query = self.query_norm(self.query_tokens).unsqueeze(0).expand(batch * 4, -1, -1)
        compressed, _ = self.compression_attention(query, source, source, need_weights=False)
        bank = self.memory_norm(compressed).reshape(batch, 4, self.memory_tokens, self.width)
        if self.collect_diagnostics:
            # Detached summaries only. Off-diagonal mean cosine uses a vector
            # sum identity, avoiding persistent attention maps or a slot Gram matrix.
            with torch.no_grad():
                raw = compressed.detach().reshape(batch, 4, self.memory_tokens, self.width)
                unit = F.normalize(bank.detach(), dim=-1, eps=1e-12)
                similarity = (unit.sum(dim=2).square().sum(dim=-1)
                              - unit.square().sum(dim=(2, 3))) / (self.memory_tokens * (self.memory_tokens - 1))
                self.last_memory_diagnostics = torch.cat((
                    raw.square().mean(dim=(0, 2, 3)).sqrt(),
                    bank.detach().square().mean(dim=(0, 2, 3)).sqrt(), similarity.mean(dim=0)))
        return bank

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
        if tuple(audio_bank.shape) != (hidden_states.shape[0], 4, self.memory_tokens, self.width):
            raise ValueError("audio memory bank must contain four aligned query32 sequences")
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
        memory = {}
        if self.last_memory_diagnostics is not None:
            values = self.last_memory_diagnostics.cpu().tolist()
            memory = {"raw_compression_rms_by_depth": values[:4],
                      "normalized_memory_rms_by_depth": values[4:8],
                      "slot_cosine_mean_by_depth": values[8:12]}
        return [{"round": index + 1, "mean_depth_weights": values[:4].cpu().tolist(),
                 "gate_mean": float(values[4]), "gate_abs_mean": float(values[5]),
                 "cross_attention_rms": float(values[6]), "residual_rms": float(values[7]),
                 "normalized_cross_attention_rms": float(values[8]),
                 "residual_token_rms_max": float(values[9]), "residual_scale": self.residual_scale,
                 "memory_tokens": self.memory_tokens, **memory}
                for index, values in self.last_diagnostics]
