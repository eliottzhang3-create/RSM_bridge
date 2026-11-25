"""Native model forwards; layer-local masks preserve position IDs and RoPE."""
import inspect
from mizar_layer_pair_search import runtime as reference
from . import config

autocast = reference.autocast
prefix = reference.prefix


def cutoff_mask(length, device, dtype, blocked_tokens=config.TOKENS):
    import torch
    if length <= blocked_tokens:
        raise ValueError('expected both prefix and text positions')
    positions = torch.arange(length, device=device)
    allowed = positions[None, :] <= positions[:, None]
    # Only text queries are isolated. Earlier audio/SEP queries retain legal
    # causal keys, so no row is entirely masked (and no all-masked-row NaN).
    allowed[blocked_tokens:, :blocked_tokens] = False
    return torch.zeros((1, 1, length, length), device=device, dtype=dtype).masked_fill_(~allowed, float('-inf'))


class AudioCutoff:
    def __init__(self, lm, *, hidden_size=576, prefix_tokens=383, blocked_tokens=127):
        if len(lm.model.layers) != 30 or len({id(layer) for layer in lm.model.layers}) != 30:
            raise ValueError('requires all thirty original distinct layers')
        self.lm, self.original = lm, tuple(lm.model.layers)
        self.hidden_size, self.prefix_tokens, self.blocked_tokens = hidden_size, prefix_tokens, blocked_tokens
        self.cutoff, self.trace, self.masked, self.forwards = None, [], [], 0
        self.enabled, self.intervention = True, None
        self._mask = None
        self.handles = [lm.register_forward_pre_hook(self._begin, with_kwargs=True),
                        lm.register_forward_hook(self._finish)]
        self.signatures = [inspect.signature(layer.forward) for layer in self.original]
        self.handles += [layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                         for i, layer in enumerate(self.original)]

    def set_cutoff(self, cutoff):
        config.condition_name(cutoff)
        if tuple(self.lm.model.layers) != self.original or self.lm.config.num_hidden_layers != 30:
            raise ValueError('layers were pruned or reordered')
        self.cutoff = cutoff

    def _begin(self, _module, args, kwargs):
        x = kwargs.get('inputs_embeds')
        if (args or x is None or kwargs.get('use_cache') is not False or kwargs.get('attention_mask') is not None
                or kwargs.get('past_key_values') is not None or x.ndim != 3 or x.shape[0] != 1
                or x.shape[2] != self.hidden_size or not self.prefix_tokens <= x.shape[1] <= 1024):
            raise ValueError('expected native fixed-prefix full-recompute forward')
        self.trace.clear()
        self.masked.clear()
        self._mask = None

    def _hook(self, index):
        def before(_module, args, kwargs):
            self.trace.append(index+1)
            if not self.enabled or self.cutoff is None or index+1 < self.cutoff:
                return None
            bound = self.signatures[index].bind_partial(*args, **kwargs)
            hidden = bound.arguments.get('hidden_states')
            if hidden is None:
                raise TypeError('unsupported decoder layer hidden-state API')
            if self.intervention is not None and index+1 == self.cutoff:
                hidden = self.intervention(hidden)
                bound.arguments['hidden_states'] = hidden
            if self._mask is None:
                self._mask = cutoff_mask(hidden.shape[1], hidden.device, hidden.dtype, self.blocked_tokens)
            bound.arguments['attention_mask'] = self._mask
            self.masked.append(index+1)
            return bound.args, bound.kwargs
        return before

    def _finish(self, _module, _args, output):
        expected = config.masked_layers(self.cutoff) if self.enabled else []
        if self.trace != list(range(1, 31)) or self.masked != expected or getattr(output, 'past_key_values', None) is not None:
            raise ValueError('actual layer/mask/cache trace differs from protocol')
        self.forwards += 1

    def evidence(self):
        return {'executed_layers_1based': list(self.trace), 'masked_layers_1based': list(self.masked),
                'cutoff_layer_1based': self.cutoff, 'blocked_audio_plus_sep_tokens': self.blocked_tokens,
                'checked_forwards': self.forwards}


def load(args, bundle, binding):
    # The existing loader strictly loads the full official model and installs
    # read-only thirty-layer audit hooks. Never call its pruning set_pair().
    loaded = reference.load(args, bundle, binding)
    loaded['cutoff_controller'] = AudioCutoff(loaded['lm'])
    import transformers
    loaded['audit'].update(experiment=config.SCHEMA, transformers=transformers.__version__,
                           attention_patch='layer_pre_hook_only', all_30_layers_retained=True)
    return loaded


def score(loaded, cached_prefix, record):
    value = reference.score(loaded, cached_prefix, record)
    return {**value, 'execution': loaded['cutoff_controller'].evidence()}


def runtime_checks(loaded, cached_prefix, record):
    """Run real numerical checks in the same allocated job, before baseline."""
    import torch
    lm, device, ctl = loaded['lm'], loaded['device'], loaded['cutoff_controller']
    ids = record['choice_token_ids'][0]
    try:
        ctl.set_cutoff(None)
        ctl.enabled = False
        native = reference.option_logits(lm, cached_prefix, ids, device)
        ctl.enabled = True
        baseline = reference.option_logits(lm, cached_prefix, ids, device)
        if not torch.equal(native, baseline):
            raise ValueError('baseline differs from native no-mask forward')
        checks = []
        for cutoff in config.ISOLATION_CHECKS:
            ctl.set_cutoff(cutoff)
            normal = reference.option_logits(lm, cached_prefix, ids, device)
            trace = ctl.evidence()
            def perturb(hidden):
                changed = hidden.clone()
                changed[:, :ctl.blocked_tokens] = 7.0
                return changed
            ctl.intervention = perturb
            isolated = reference.option_logits(lm, cached_prefix, ids, device)
            torch.testing.assert_close(normal, isolated, rtol=0, atol=0)
            checks.append({'cutoff_layer_1based': cutoff, 'status': 'PASS', 'execution': trace,
                           'audio_sep_intervention_max_abs_error': float((normal-isolated).abs().max().item())})
            ctl.intervention = None
        return {'status': 'PASS', 'baseline_native_bitwise_equal': True, 'cutoff_isolation': checks,
                'position_policy': 'native_positions_unmodified', 'all_30_layers': True}
    finally:
        ctl.intervention, ctl.enabled = None, True
        ctl.set_cutoff(None)
