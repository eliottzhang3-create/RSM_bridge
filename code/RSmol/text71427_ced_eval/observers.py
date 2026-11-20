"""Read-only hooks: retain metadata, never activations or patched forwards."""
from .registry import SCHEDULE, variant_spec

TARGETS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
           'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def expected_routers(variant, length, width):
    events = []
    for index, boundary in enumerate((7, 21, 35)):
        def add(name, tokens, slots):
            events.append((boundary, name, (1, tokens, width), (1, tokens, slots)))
        if variant == 'audio7_read2_ar6':
            add(f'audio_write{index}', 126, 5)
            add(f'write{index}', length - 126, 5)
            add(f'audio_read{index}', 126, 7 if index < 2 else 5)
            add(f'read{index}', length - 126, 5)
        elif variant == 'audio7_read2' and index < 2:
            add(f'write{index}', length, 5)
            add(f'audio_read{index}', 126, 7)
            add(f'read{index}', length - 126, 5)
        else:
            add(f'write{index}', length, 5)
            add(f'read{index}', length, 5)
            if variant == 'listen' and index < 2:
                add(f'listen{4 if index == 0 else 8}', 126, 5)
    return events


def install_trace_audit(lm, variant):
    spec, core = variant_spec(variant), lm.model
    trace, routers, branches, projections = [], [], [], []
    lora_modules = []
    audit = {'status': 'PENDING', 'checked_forwards': 0, 'expected_schedule': list(SCHEDULE),
             'memory_policy': 'fresh_per_full_sequence_forward', 'use_cache': False,
             'audio_memory_slots': 7 if variant.startswith('audio7') else 5,
             'text_memory_slots': 5, 'router_modules': spec['router_modules']}
    width = lm.config.hidden_size

    def idle():
        if core.gradient_checkpointing or any(m.loop_index is not None for _, _, m in lora_modules):
            raise RuntimeError('LoRA scope not reset or gradient checkpointing enabled')

    def begin(_module, _inputs, kwargs):
        trace.clear(); routers.clear(); branches.clear(); projections.clear()
        idle()
        if (kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None
                or kwargs.get('attention_mask') is not None or kwargs.get('labels') is not None
                or kwargs.get('input_ids') is not None or kwargs.get('inputs_embeds') is None):
            raise RuntimeError('expected full inputs_embeds, no KV cache, labels or attention mask')

    def physical(index):
        def hook(_module, _inputs, _output):
            trace.append(index)
        return hook

    def router(name):
        def hook(_module, inputs, output):
            routers.append((len(trace), name, tuple(inputs[0].shape), tuple(output.shape)))
        return hook

    def branch(layer, family):
        def hook(_module, inputs, output):
            branches.append((len(trace), layer, family, tuple(inputs[0].shape), tuple(output.shape)))
        return hook

    def projection(physical_index, target):
        def hook(module, inputs):
            logical = len(trace)
            loop = (logical - 7) // 14 if 7 <= logical < 35 else None
            if loop is None or physical_index != SCHEDULE[logical] or module.loop_index != loop:
                raise RuntimeError('LoRA projection selected the wrong physical layer or loop')
            projections.append((logical, physical_index, target, loop, tuple(inputs[0].shape)))
        return hook

    def finish(_module, _inputs, output):
        idle()
        if trace != list(SCHEDULE) or output.past_key_values is not None:
            raise RuntimeError('actual 42-call physical schedule or cache policy differs')
        if [x['physical_index'] for x in core.last_forward_trace] != trace:
            raise RuntimeError('internal and observed physical traces disagree')
        length = int(output.logits.shape[1])
        if routers != expected_routers(variant, length, width):
            raise RuntimeError(f'router order/shape differs: {routers}')
        expected_branch = []
        if variant in ('loop_lora', 'listen', 'wfuse'):
            family = 'fusion' if variant == 'wfuse' else 'gate'
            expected_branch = [(boundary, layer, family,
                                (1, 126, width * (2 if family == 'fusion' else 1)),
                                (1, 126, width if family == 'fusion' else 1))
                               for boundary, layer in ((7, 4), (21, 8))]
        if branches != expected_branch:
            raise RuntimeError('CED branch placement or audio-only dimensions differ')
        if variant != 'base':
            expected = [(logical, SCHEDULE[logical], target, (logical - 7) // 14,
                         (1, length, module.in_features))
                        for logical in range(7, 35)
                        for physical_index, target, module in lora_modules
                        if physical_index == SCHEDULE[logical]]
            if sorted(projections) != sorted(expected) or len(projections) != 196:
                raise RuntimeError('expected 98 LoRA projection calls per loop')
            expected_trace = [(i, SCHEDULE[i], (i - 7) // 14) for i in range(7, 35)]
            if list(core.last_loop_lora_trace) != expected_trace:
                raise RuntimeError('internal and observed LoRA traces disagree')
            audit.update(loop_lora_projection_calls=196, loop_lora_calls_per_loop=[98, 98],
                         loop_lora_indices_reset=True)
        audit.update(status='PASS', checked_forwards=audit['checked_forwards'] + 1,
                     last_sequence_length=length, last_schedule=list(trace),
                     last_router_events=[list(x[:2]) + [list(x[2]), list(x[3])] for x in routers],
                     last_branch_events=[list(x[:3]) + [list(x[3]), list(x[4])] for x in branches])

    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, layer in enumerate(core.layers):
        layer.register_forward_hook(physical(index))
    for kind in ('write', 'read', 'audio_write', 'audio_read'):
        for index, module in enumerate(getattr(core, kind + '_routers', [])):
            module.register_forward_hook(router(f'{kind}{index}'))
    if variant == 'listen':
        for layer, module in core.listen_routers.items():
            module.register_forward_hook(router('listen' + layer))
    if variant in ('loop_lora', 'listen', 'wfuse'):
        family = 'fusion' if variant == 'wfuse' else 'gate'
        for layer, module in (core.audio_fusions if variant == 'wfuse' else core.audio_gates).items():
            module.register_forward_hook(branch(int(layer), family))
    if variant != 'base':
        for index in range(7, 21):
            for target in TARGETS:
                module = core.layers[index].get_submodule(target)
                lora_modules.append((index, target, module))
                module.register_forward_pre_hook(projection(index, target))
    lm.register_forward_hook(finish)
    return audit
