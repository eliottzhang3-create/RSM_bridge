"""Input reinjection then tokenwise CED4/8/12 audio CA, shared at both loops."""
import torch
from torch import nn


class MultilayerAudioCrossAttention(nn.Module):
    def __init__(self, width=576, num_heads=9, audio_tokens=126, prompt_tokens=256):
        super().__init__()
        if width % num_heads:
            raise ValueError("cross attention width must be divisible by num_heads")
        self.width, self.audio_tokens, self.prompt_tokens = width, audio_tokens, prompt_tokens
        self.text_start = audio_tokens + 1
        # Register after strict text loading; preserve mapper/LM and caller RNG.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleList([
                nn.Linear(1536, width, device="cpu", dtype=torch.float32) for _ in range(3)])
            self.input_projection = nn.Linear(width, width, device="cpu", dtype=torch.float32)
            self.input_gate = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
            self.gate1 = nn.Linear(width, 3, device="cpu", dtype=torch.float32)
            self.gate2 = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
            self.cross_attention = nn.MultiheadAttention(
                width, num_heads, dropout=0., bias=True, batch_first=True,
                device="cpu", dtype=torch.float32)
            for projection in self.projections:
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            # Input projection retains default Linear weight initialization.
            nn.init.zeros_(self.input_projection.bias)
            for gate in (self.input_gate, self.gate1, self.gate2):
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            nn.init.zeros_(self.cross_attention.in_proj_bias)
            nn.init.zeros_(self.cross_attention.out_proj.bias)
            self.audio_input_norms = nn.ModuleList([
                nn.LayerNorm(1536, eps=1e-5, device="cpu", dtype=torch.float32) for _ in range(3)])
            self.audio_output_norms = nn.ModuleList([
                nn.LayerNorm(width, eps=1e-5, device="cpu", dtype=torch.float32) for _ in range(3)])
            self.query_norm = nn.LayerNorm(width, eps=1e-5, device="cpu", dtype=torch.float32)
        self.collect_diagnostics = False
        self.last_diagnostics = []
        self.last_injection_trace = ()

    def project_audio(self, grids, hidden_states):
        expected = (hidden_states.shape[0], 3, self.audio_tokens, 1536)
        if grids is None or tuple(grids.shape) != expected:
            raise ValueError(f"CED three-depth grids must have shape {expected}")
        if grids.device != hidden_states.device or grids.dtype != hidden_states.dtype:
            raise ValueError("CED grids and LM inputs must share FP32 dtype/device")
        self.last_diagnostics = []
        self.last_injection_trace = ()
        return torch.stack([self.audio_output_norms[index](
                                projection(self.audio_input_norms[index](grids[:, index])))
                            for index, projection in enumerate(self.projections)], dim=1)

    def project_input(self, inputs_embeds):
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] < self.audio_tokens + 1 or inputs_embeds.shape[2] != self.width:
            raise ValueError("input reinjection requires original audio126+SEP embeddings")
        # Keep the mapper graph; this projected bank is reused by both loops.
        return self.input_projection(inputs_embeds[:, :self.audio_tokens])

    def forward(self, hidden_states, audio_bank, input_residual, round_index):
        if round_index not in (0, 1):
            raise ValueError("only the two middle-loop entrances receive cross attention")
        if (hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width
                or hidden_states.shape[1] < self.audio_tokens + 1):
            raise ValueError("invalid cross attention hidden shape")
        expected = (hidden_states.shape[0], 3, self.audio_tokens, self.width)
        if tuple(audio_bank.shape) != expected or tuple(input_residual.shape) != (expected[0], self.audio_tokens, self.width):
            raise ValueError("invalid prepared audio/input bank shape")
        audio = hidden_states[:, :self.audio_tokens]
        input_gate = torch.tanh(self.input_gate(audio))
        input_delta = input_gate * input_residual
        injected = audio + input_delta
        weights = torch.softmax(self.gate1(injected), dim=-1)
        mixed = (audio_bank * weights.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
        query = self.query_norm(injected)
        delta, _ = self.cross_attention(query, mixed, mixed, need_weights=False)
        ca_gate = torch.tanh(self.gate2(injected))
        ca_delta = ca_gate * delta
        updated = injected + ca_delta
        self.last_injection_trace += ((round_index, "input", "cross_attention"),)
        if self.collect_diagnostics:
            # Only detached small scalars survive; never retain a training graph.
            stats = torch.cat((weights.detach().mean(dim=(0, 1)), torch.stack((
                input_gate.detach().mean(), input_gate.detach().abs().mean(),
                ca_gate.detach().mean(), ca_gate.detach().abs().mean(),
                input_delta.detach().square().mean().sqrt(),
                delta.detach().square().mean().sqrt(),
                ca_delta.detach().square().mean().sqrt(),
                audio.detach().square().mean().sqrt(),
                ca_delta.detach().square().mean(dim=-1).sqrt().max()))))
            self.last_diagnostics.append((round_index, stats))
        return torch.cat((updated, hidden_states[:, self.audio_tokens:]), dim=1)

    def diagnostics(self):
        return [{"round": index + 1, "mean_depth_weights": values[:3].cpu().tolist(),
                 "input_gate_mean": float(values[3]), "input_gate_abs_mean": float(values[4]),
                 "ca_gate_mean": float(values[5]), "ca_gate_abs_mean": float(values[6]),
                 "input_residual_rms": float(values[7]), "cross_attention_rms": float(values[8]),
                 "ca_residual_rms": float(values[9]), "audio_hidden_rms": float(values[10]),
                 "ca_residual_token_rms_max": float(values[11])}
                for index, values in self.last_diagnostics]
