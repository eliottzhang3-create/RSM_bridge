"""Prompt-selected CED depths written into the existing MeSH audio positions.

Only the added branch is normalized. Original MeSH states/routers are untouched.
The two zero-initialized scalar amplitudes make both initial writes exactly zero.
"""
import math

import torch
from torch import nn


class MultilayerAudioListen(nn.Module):
    def __init__(self, width=576, audio_tokens=126, prompt_tokens=256):
        super().__init__()
        self.width, self.audio_tokens, self.prompt_tokens = width, audio_tokens, prompt_tokens
        self.text_start = audio_tokens + 1
        self.gamma = 0.3
        self.memory_slots = 5
        # Preserve the mother's mapper/LM initialization and subsequent RNG draws.
        # This module is registered only AFTER strict loading of all original LM weights.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleList([
                nn.Linear(1536, width, device="cpu", dtype=torch.float32) for _ in range(4)])
            self.gates = nn.ModuleList([
                nn.Linear(width, 4, device="cpu", dtype=torch.float32) for _ in range(2)])
            self.listen_routers = nn.ModuleList([
                nn.Linear(width, self.memory_slots, device="cpu", dtype=torch.float32) for _ in range(2)])
            self.amplitudes = nn.ParameterList([
                nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in range(2)])
            for projection in self.projections:
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            for router in self.listen_routers:
                # Exact original MeSH router policy, independently sampled per round.
                std = math.sqrt(2.0 / (self.memory_slots * width))
                with torch.no_grad():
                    values = torch.empty_like(router.weight)
                    while True:
                        values.normal_(0.0, std)
                        values.clamp_(-3.0 * std, 3.0 * std)
                        if torch.isfinite(values).all():
                            break
                    router.weight.copy_(values)
                    router.bias.zero_()
            self.audio_input_norms = nn.ModuleList([
                nn.LayerNorm(1536, eps=1e-5, elementwise_affine=False) for _ in range(4)])
            self.audio_output_norms = nn.ModuleList([
                nn.LayerNorm(width, eps=1e-5, elementwise_affine=False) for _ in range(4)])
            self.text_norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
            self.query_norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
        self.gradient_audit_state = None
        self.collect_diagnostics = False
        self.last_diagnostics = []
        self.last_rounds = []
        self.last_gradient_diagnostics = None

    def project_audio(self, grids, hidden_states, prompt_mask):
        expected = (hidden_states.shape[0], 4, self.audio_tokens, 1536)
        if grids is None or tuple(grids.shape) != expected:
            raise ValueError(f"CED listen grids must have shape {expected}")
        if grids.dtype != hidden_states.dtype or grids.device != hidden_states.device:
            raise ValueError("CED grids and LM inputs must share FP32 dtype/device")
        if hidden_states.dtype != torch.float32:
            raise ValueError("CED listen requires the mother's FP32 forward")
        if hidden_states.shape[1] < self.text_start + self.prompt_tokens:
            raise ValueError("listen requires the complete audio/SEP/fixed256 prompt")
        self.validate_prompt_mask(prompt_mask, hidden_states.shape[0])
        self.last_diagnostics = []
        self.last_rounds = []
        return torch.stack([
            self.audio_output_norms[i](projection(self.audio_input_norms[i](grids[:, i])))
            for i, projection in enumerate(self.projections)], dim=1)

    def validate_prompt_mask(self, mask, batch_size):
        if mask is None or tuple(mask.shape) != (batch_size, self.prompt_tokens):
            raise ValueError("listen requires the tokenizer's fixed256 prompt attention_mask")
        if not bool((((mask == 0) | (mask == 1)).all() & (mask.sum(dim=1) > 0).all()).item()):
            raise ValueError("prompt mask must be binary with at least one valid token per sample")

    def pool_prompt(self, hidden_states, prompt_mask):
        # This side-channel mask is NEVER passed to the original LM attention.
        prompt = self.text_norm(hidden_states[:, self.text_start:self.text_start + self.prompt_tokens])
        mask = prompt_mask.to(device=prompt.device, dtype=prompt.dtype).unsqueeze(-1)
        return (prompt * mask).sum(dim=1) / mask.sum(dim=1)

    def forward(self, memory, boundary_hidden, audio_bank, prompt_mask, round_index, read_weights):
        if round_index not in (0, 1):
            raise ValueError("listen writes only before the first and second middle loops")
        if tuple(memory.shape) != (boundary_hidden.shape[0], 5, boundary_hidden.shape[1], self.width):
            raise ValueError("listen must update the existing five-slot MeSH memory")
        weights = torch.sigmoid(self.gates[round_index](self.pool_prompt(boundary_hidden, prompt_mask)))
        audio = (audio_bank * weights[:, :, None, None]).sum(dim=1)
        query = self.query_norm(boundary_hidden[:, :self.audio_tokens])
        probabilities = torch.softmax(self.listen_routers[round_index](query).float(), dim=-1).to(query.dtype)
        amplitude = torch.tanh(self.amplitudes[round_index])
        # RMS(a_i)<=1, sum(sigmoid_i)<=4, |tanh(beta)|<=1, p_slot<=1.
        # Thus every added slot/token RMS <= gamma (finite arithmetic assumed).
        delta = (self.gamma / 4.0) * amplitude * audio.unsqueeze(1) * probabilities.transpose(1, 2).unsqueeze(-1)
        self.last_rounds.append(round_index)
        if self.collect_diagnostics:
            with torch.no_grad():
                reads = read_weights[:, :self.audio_tokens].detach().transpose(1, 2).unsqueeze(-1)
                direct_read = (delta.detach() * reads).sum(dim=1)
                base_read = (memory[:, :, :self.audio_tokens].detach() * reads).sum(dim=1)
                base_rms = base_read.square().mean(dim=-1).sqrt()
                read_rms = direct_read.square().mean(dim=-1).sqrt()
                p = probabilities.detach()
                stats = torch.cat((weights.detach().mean(dim=0), p.mean(dim=(0, 1)),
                    torch.stack((-(p.clamp_min(1e-12).log() * p).sum(dim=-1).mean(),
                                 self.amplitudes[round_index].detach()[0], amplitude.detach()[0],
                                 audio.detach().square().mean().sqrt(),
                                 delta.detach().square().mean(dim=-1).sqrt().max(),
                                 read_rms.max(), (read_rms / base_rms.clamp_min(1e-8)).max(),
                                 ((weights.detach() < .01) | (weights.detach() > .99)).float().mean(),
                                 base_rms.mean(),
                                 boundary_hidden[:, :self.audio_tokens].detach().square().mean().sqrt()))))
                self.last_diagnostics.append((round_index, stats))
        # No in-place mutation of memory views needed by autograd. SEP/text are identical.
        return torch.cat((memory[:, :, :self.audio_tokens] + delta,
                          memory[:, :, self.audio_tokens:]), dim=2)

    def diagnostics(self):
        result = []
        for index, tensor in self.last_diagnostics:
            values = tensor.cpu().tolist()
            result.append({"round": index + 1, "mean_depth_sigmoids": values[:4],
                           "mean_slot_probabilities": values[4:9], "slot_entropy": values[9],
                           "beta": values[10], "tanh_beta": values[11], "aggregate_audio_rms": values[12],
                           "write_slot_token_rms_max": values[13], "direct_read_token_rms_max": values[14],
                           "direct_read_to_base_rms_ratio_max": values[15], "gamma": self.gamma,
                           "depth_gate_saturation_fraction": values[16],
                           "base_read_token_rms_mean": values[17], "boundary_audio_rms": values[18],
                           "gradient_snapshot": self.last_gradient_diagnostics,
                           "scope": "last_microbatch_before_optimizer_update"})
        return result

    def capture_gradient_diagnostics(self, update):
        """Read unscaled gradients before the unchanged mother's GradNorm clipping."""
        branches = {f"projection_{i}": list(m.parameters()) for i, m in enumerate(self.projections)}
        branches.update({f"gate_{i}": list(m.parameters()) for i, m in enumerate(self.gates)})
        branches.update({f"router_{i}": list(m.parameters()) for i, m in enumerate(self.listen_routers)})
        branches.update({f"amplitude_{i}": [p] for i, p in enumerate(self.amplitudes)})
        values = {}
        with torch.no_grad():
            for name, parameters in branches.items():
                grads = [p.grad.detach() for p in parameters if p.grad is not None]
                values[name] = {"present": len(grads) == len(parameters),
                    "l2": float(torch.stack([g.double().square().sum() for g in grads]).sum().sqrt().item())
                          if grads else None,
                    "finite": all(bool(torch.isfinite(g).all().item()) for g in grads) and bool(grads)}
        self.last_gradient_diagnostics = {"update": update,
            "scope": "rank0_accumulated_unscaled_before_GradNorm", "modules": values}
