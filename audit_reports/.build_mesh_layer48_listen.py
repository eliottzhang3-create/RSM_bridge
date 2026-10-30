from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine'
MOTHER = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small'
SOURCE = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject'
DIRECT = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject'
FOUR = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen'
DEST = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen'
assert not (DEST/'models/recursive_mesh_listen.py').exists(), DEST

def read(path):
    return path.read_text(encoding='utf-8-sig')

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')

def change(text, old, new, count=1):
    assert text.count(old) == count, (old[:120], text.count(old), count)
    return text.replace(old, new)

protected = {}
for directory in (MOTHER, SOURCE, DIRECT, FOUR):
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.json'} and '__pycache__' not in path.parts:
            protected[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
for path in (ROOT/'code/RSmol/recursive_model_5_10x2_5_mesh.py', ROOT/'code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh', ROOT/'code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_5090.sh'):
    protected[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
write(ROOT/'audit_reports/mizar_s1_mesh_ced_layer48_listen_parent_baseline.json', json.dumps(protected, indent=2)+'\n')
for path in MOTHER.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        if path.suffix in {'.py', '.sh', '.md'} or path.name in {'.gitignore', 'LICENSE', 'NOTICE'}:
            write(DEST/path.relative_to(MOTHER), read(path))

# Exact audio feature extraction, intermediate MLPs and explicit A4/A8 plumbing.
for name in ('models/ced.py', 'models/mellow.py'):
    write(DEST/name, read(SOURCE/name))
write(DEST/'models/decoder.py', read(DIRECT/'models/decoder.py'))

text = read(ROOT/'code/RSmol/recursive_model_5_10x2_5_mesh.py')
text = change(text, '    def _route(self,', '''    def enable_audio_listen(self, audio_tokens: int = 126) -> None:
        """Register fresh modules only after strict loading of the six original routers."""
        if hasattr(self, "audio_gates"):
            raise RuntimeError("refusing to reinitialize audio listen modules")
        if audio_tokens != 126:
            raise ValueError("CED listen requires 126 audio tokens")
        weight = self.embed_tokens.weight
        with torch.random.fork_rng(devices=[]):
            gates = nn.ModuleDict({str(layer): nn.Linear(self.config.hidden_size, 1,
                device="cpu", dtype=weight.dtype) for layer in (4, 8)})
            routers = nn.ModuleDict({str(layer): nn.Linear(self.config.hidden_size, MEMORY_SLOT_COUNT,
                device="cpu", dtype=weight.dtype) for layer in (4, 8)})
            for gate in gates.values():
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            for router in routers.values():
                _init_router(router)
        self.audio_gates = gates.to(weight.device)
        self.listen_routers = routers.to(weight.device)
        self.listen_audio_tokens = audio_tokens
        self.collect_listen_stats = False
        self.last_listen_trace = ()
        self.last_listen_stats = []
        self.listen_gradient_state = None
        self.config.use_cache = False

    def listen_audio(self, memory, boundary_hidden, audio_information, ced_layer, logical_index):
        """After original write, add gated CED values to the same audio positions in five slots."""
        audio = boundary_hidden[:, :self.listen_audio_tokens]
        gate = torch.tanh(self.audio_gates[str(ced_layer)](audio))
        probabilities = self._route(self.listen_routers[str(ced_layer)], audio,
                                    "listen_pre" if ced_layer == 4 else "listen_0")
        value = gate * audio_information
        delta = value.unsqueeze(1) * probabilities.transpose(1, 2).unsqueeze(-1)
        if self.collect_listen_stats:
            with torch.no_grad():
                absolute = gate.detach().abs()
                p = probabilities.detach().float()
                stats = torch.cat((torch.stack((absolute.mean(), absolute.max(),
                    (absolute > .95).float().mean(), value.detach().square().mean().sqrt(),
                    delta.detach().square().mean().sqrt(),
                    delta.detach().square().mean(dim=-1).sqrt().max(),
                    -(p.clamp_min(1e-12).log() * p).sum(dim=-1).mean())), p.mean(dim=(0, 1))))
                self.last_listen_stats.append((logical_index, ced_layer, stats))
        self.last_listen_trace += ((logical_index, ced_layer),)
        # Out-of-place: keep autograd's saved memory views valid; SEP/text are unchanged.
        return torch.cat((memory[:, :, :self.listen_audio_tokens] + delta,
                          memory[:, :, self.listen_audio_tokens:]), dim=2)

    def listen_diagnostics(self):
        result = []
        for logical_index, layer, tensor in self.last_listen_stats:
            values = tensor.cpu().tolist()
            result.append({"logical_index": logical_index, "ced_layer": layer,
                "mean_abs_gate": values[0], "max_abs_gate": values[1],
                "gate_saturation_fraction": values[2], "gated_audio_rms": values[3],
                "memory_delta_rms": values[4], "memory_delta_token_rms_max": values[5],
                "listen_entropy": values[6], "mean_slot_probabilities": values[7:],
                "scope": "rank0_last_microbatch"})
        return result

    def _route(self,''')
text = change(text, 'cache_position: torch.LongTensor | None = None, **kwargs: Any)',
              'cache_position: torch.LongTensor | None = None, audio_information=None, **kwargs: Any)')
text = change(text, '        output_attentions = bool(_cfg', '''        if not hasattr(self, "audio_gates"):
            raise RuntimeError("strict-load the backbone, then enable_audio_listen before forward")
        if inputs_embeds is None or input_ids is not None or use_cache or past_key_values is not None:
            raise ValueError("audio listen requires full-prefix inputs_embeds and use_cache=False")
        if attention_mask is not None:
            raise ValueError("audio listen retains the mother's attention_mask=None")
        if inputs_embeds.shape[1] < self.listen_audio_tokens:
            raise ValueError("audio listen requires the complete 126-token audio prefix")
        expected = (inputs_embeds.shape[0], self.listen_audio_tokens, self.config.hidden_size)
        if not isinstance(audio_information, (tuple, list)) or len(audio_information) != 2:
            raise ValueError("explicit (A4,A8) audio_information is required for every forward")
        for value in audio_information:
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
                    or value.dtype != inputs_embeds.dtype or value.device != inputs_embeds.device):
                raise ValueError("A4/A8 shape, dtype and device must match the audio prefix")
        self.last_listen_trace = ()
        self.last_listen_stats = []
        output_attentions = bool(_cfg''')
text = change(text, '        memory = self._write(memory, prefix_output, write_pre)', '''        memory = self._write(memory, prefix_output, write_pre)
        memory = self.listen_audio(memory, prefix_output, audio_information[0], 4, 5)''')
text = change(text, '            memory = self._write(memory, core, write)', '''            memory = self._write(memory, core, write)
            if loop == 0:
                memory = self.listen_audio(memory, core, audio_information[1], 8, 15)''')
text = change(text, 'logits_to_keep: int | torch.Tensor = 0, **kwargs: Any)',
              'logits_to_keep: int | torch.Tensor = 0, audio_information=None, **kwargs: Any)')
text = change(text, 'return_dict=True, cache_position=cache_position)', 'return_dict=True, cache_position=cache_position, audio_information=audio_information)')
write(DEST/'models/recursive_mesh_listen.py', text)

text = read(DIRECT/'models/recursive_text.py').replace('models.recursive_layer_inject', 'models.recursive_mesh_listen')
text = text.replace('enable_layer_injection', 'enable_audio_listen').replace('validate_layer_injection', 'validate_audio_listen')
text = change(text, 'core.layer_inject_tokens != 126 or core.layer_inject_indices != (5, 15)', 'core.listen_audio_tokens != 126')
text = change(text, '    return {"ced_layers":', '''    routers = list(core.write_routers) + list(core.read_routers) + list(core.listen_routers.values())
    if (set(core.listen_routers) != {"4", "8"} or len({id(r) for r in routers}) != 8
            or any(tuple(r.weight.shape) != (5, width) or tuple(r.bias.shape) != (5,)
                   for r in core.listen_routers.values())):
        raise ValueError("listen requires two independent Linear(hidden_size,5) routers")
    if sum(p.numel() for p in core.listen_routers.parameters()) != 2 * (width * 5 + 5):
        raise ValueError("listen router parameter count mismatch")
    return {"ced_layers":''')
text = text.replace('after_memory_read_before_middle_loop', 'after_original_write_before_original_read')
text = change(text, '"gate_parameter_count": 2 * (width + 1)', '''"gate_parameter_count": 2 * (width + 1),
            "listen_router_parameter_count": 2 * (width * 5 + 5),
            "listen_query": "module_output_audio_positions", "original_routers": 6, "total_routers": 8''')
write(DEST/'models/recursive_text.py', text)

text = read(DIRECT/'optim.py')
text = change(text, '    if name.startswith("audio_encoder.base.mapper."):', '''        if name.startswith(f"caption_decoder.lm.model.listen_routers.{layer}."):
            return f"layer{layer}_listen"
    if name.startswith("audio_encoder.base.mapper."):''')
text = change(text, '"layer4_gate", "layer8_gate"}', '"layer4_gate", "layer8_gate", "layer4_listen", "layer8_listen"}')
text = text.replace('two independent MLPs and two gates', 'two MLPs, two gates and two listen routers')
write(DEST/'optim.py', text)

text = read(DIRECT/'contract.py').replace('recursive_mesh_layer4_8_inject_v1', 'recursive_mesh_layer4_8_listen_v1')
text = text.replace('INJECTION_CONTRACT', 'LISTEN_CONTRACT').replace('"layer_injection"', '"audio_listen"')
text = text.replace('after_memory_read_before_middle_loop', 'after_original_write_before_original_read')
text = change(text, 'models/recursive_layer_inject.py', 'models/recursive_mesh_listen.py')
text = change(text, '    "max_lr": 6e-4,', '''    "listen_router": "Linear_576_5_bias_softmax_slots", "independent_listen_routers": True,
    "gate_and_listen_query": "module_output_audio_positions",
    "original_loop_router_query": "loop_input", "original_loop_write_value": "loop_output",
    "listen_router_init": "normal_sqrt_2_over_5H_clamp_3sigma_bias_zero",
    "memory_update": "M_after_write_plus_slot_probability_times_tanh_gate_times_A",
    "new_parameter_count": 2450700,
    "max_lr": 6e-4,''')
resume = read(FOUR/'contract.py')
resume = resume[resume.index('def validate_resume_payload('):]
resume = change(resume, '''    expected_branches = {f"{kind}_{i}" for kind, count in
                         (("projection", 4), ("gate", 2), ("router", 2), ("amplitude", 2)) for i in range(count)}''', '''    expected_branches = {f"layer{layer}_{kind}" for layer in (4, 8) for kind in ("mlp", "gate", "listen")}''')
text += '\n\n'+resume
write(DEST/'contract.py', text)
text = read(DEST/'write_config.py')
text = change(text, '    contract = semantic_contract', '    contract = semantic_contract')
text = change(text, '    config = {', '    config = {')
text = change(text, '              "intentional_differences": [', '''              "intentional_differences": [
                  "CED block4/block8 independent MLPs; independent zero tanh gates and softmax listen routers write existing audio memory after original writes0/1 and before reads0/1; new maxLR6e-4",''')
write(DEST/'write_config.py', text)

launcher = ROOT/'code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh'
text = read(launcher).replace('recursive_mesh_ced_small', 'recursive_mesh_ced_small_layer4_8_listen')
text = text.replace('recursive_mesh_5090', 'recursive_mesh_layer4_8_listen_5090').replace('mizar-s1-ced-mesh-gbs256-', 'mizar-ced-m48listen-')
write(ROOT/'code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_listen_5090.sh', text)
text = read(DEST/'run.sh').replace('recursive_mesh_ced_small', 'recursive_mesh_ced_small_layer4_8_listen')
write(DEST/'run.sh', text)
print(DEST)
