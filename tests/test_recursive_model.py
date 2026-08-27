"""Dependency-light local tests for the strict recursive Llama fixture."""

from __future__ import annotations

import json
import inspect

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from code.RSmol.recursive_model import (  # noqa: E402
    MAPPING_POLICY,
    RecursiveLlamaForCausalLM,
    _validate_cache_capacity,
    build_stepwise_mapping,
    make_dynamic_cache,
    parameter_audit,
    register_auto_class,
)


def tiny_model() -> RecursiveLlamaForCausalLM:
    config = transformers.LlamaConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=64,
        # ``num_hidden_layers`` is logical depth/cache slots.  The model owns
        # only ``recursive_layer_count`` unique physical decoder modules.
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )
    config.recursive_loops = 2
    config.recursive_layer_count = 2
    config.recursive_mapping_policy = "explicit_fixture"
    config.recursive_source_layer_indices_0based = [0, 3]
    config.recursive_source_layer_indices_1based = [1, 4]
    return RecursiveLlamaForCausalLM(config)


def test_mapping_is_explicit_and_rejects_implicit_future_depth() -> None:
    mapping = build_stepwise_mapping(30)
    assert MAPPING_POLICY == "explicit_1based_odd_plus_last"
    assert mapping == tuple(range(0, 27, 2)) + (29,)
    assert [index + 1 for index in mapping] == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 30]
    with pytest.raises(ValueError, match="No implicit Stepwise mapping"):
        build_stepwise_mapping(4)
    assert build_stepwise_mapping(4, source_layer_indices_0based=[0, 3]) == (0, 3)


def test_config_requires_logical_and_physical_depth_contract() -> None:
    config = tiny_model().config
    delattr(config, "recursive_layer_count")
    with pytest.raises(ValueError, match="missing recursive_layer_count"):
        RecursiveLlamaForCausalLM(config)
    config = tiny_model().config
    config.recursive_layer_count = 3
    with pytest.raises(ValueError, match="depth mismatch"):
        RecursiveLlamaForCausalLM(config)


def test_shared_stack_state_dict_and_cache_slots() -> None:
    model = tiny_model().eval()
    assert len(model.model.layers) == 2
    calls: list[int] = []
    parameter_calls: list[int] = []

    def record_physical_use(module, *_):
        calls.append(id(module))
        parameter_calls.append(id(module.self_attn.q_proj.weight))

    hook = model.model.layers[0].register_forward_hook(record_physical_use)
    assert not any("model.layers.2." in key for key in model.state_dict())
    layer_indices = {
        int(key.split("model.layers.", 1)[1].split(".", 1)[0])
        for key in model.state_dict()
        if key.startswith("model.layers.")
    }
    assert layer_indices == {0, 1}
    audit = parameter_audit(model)
    assert audit["logical_layer_count"] == 4
    assert audit["physical_layer_count"] == 2
    assert audit["recursive_layer_count"] == 2
    assert audit["logical_cache_slot_count"] == 4
    assert audit["depth_consistent"] is True
    assert audit["recursive_shared_parameter_count_logical_references"] == (
        audit["recursive_shared_parameter_count_unique"] * 2
    )
    input_ids = torch.tensor([[3, 4, 5, 6]])
    no_cache = model(input_ids=input_ids, use_cache=False)
    hook.remove()
    assert calls == [id(model.model.layers[0]), id(model.model.layers[0])], (
        "the same physical layer object must execute once per recursive loop"
    )
    assert parameter_calls[0] == parameter_calls[1]
    assert sum(parameter is model.model.layers[0].self_attn.q_proj.weight for parameter in model.parameters()) == 1
    cached = model(input_ids=input_ids, use_cache=True)
    assert no_cache.logits.shape == (1, 4, 31)
    assert torch.allclose(no_cache.logits, cached.logits, atol=1e-5, rtol=1e-4)
    assert len(cached.past_key_values) >= 4
    assert all(cached.past_key_values.get_seq_length(index) == 4 for index in range(4))
    if hasattr(cached.past_key_values, "layers"):
        assert all(cached.past_key_values.layers[index].keys.numel() > 0 for index in range(4))
        assert all(cached.past_key_values.layers[index].values.numel() > 0 for index in range(4))
    else:
        assert all(cached.past_key_values.key_cache[index].numel() > 0 for index in range(4))
        assert all(cached.past_key_values.value_cache[index].numel() > 0 for index in range(4))
    class FixedCapacityCache:
        def __len__(self) -> int:
            return 2

        def update(self, *args, **kwargs):
            raise AssertionError("capacity validation must fail before update")

        def get_seq_length(self, *args, **kwargs) -> int:
            return 0

    with pytest.raises(ValueError, match="required_slots=4"):
        model(input_ids=input_ids, past_key_values=FixedCapacityCache(), use_cache=True)

    lazy_cache = make_dynamic_cache()
    assert len(lazy_cache) == 0
    _validate_cache_capacity(lazy_cache, logical_layer_count=4)


def test_prefill_incremental_labels_backward_and_update() -> None:
    model = tiny_model()
    model.eval()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.ones_like(input_ids)
    prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    incremental = model(
        input_ids=input_ids[:, -1:],
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )
    full = model(input_ids=torch.cat((input_ids, input_ids[:, -1:]), dim=1), use_cache=False)
    assert torch.allclose(incremental.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-4)

    model.train()
    labels = input_ids.clone()
    result = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert result.loss is not None and torch.isfinite(result.loss)
    expected_loss = torch.nn.functional.cross_entropy(
        result.logits[:, :-1, :].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
    )
    assert torch.allclose(result.loss, expected_loss, atol=1e-6, rtol=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    backward_calls: list[int] = []
    backward_hook = model.model.layers[0].register_full_backward_hook(
        lambda module, grad_input, grad_output: backward_calls.append(id(module))
    )
    result.loss.backward()
    backward_hook.remove()
    assert backward_calls == [id(model.model.layers[0]), id(model.model.layers[0])]
    gradient = model.model.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    optimizer.step()
    assert not torch.equal(before, model.model.layers[0].self_attn.q_proj.weight.detach())


def test_standard_model_output_controls_and_explicit_unsupported_kwargs() -> None:
    model = tiny_model().eval()
    input_ids = torch.tensor([[3, 4, 5]])
    if "output_attentions" in inspect.signature(model.model.layers[0].forward).parameters:
        result = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
        )
    else:
        with pytest.raises(RuntimeError, match="does not expose output_attentions"):
            model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )
        result = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    assert result.hidden_states is not None and len(result.hidden_states) == len(model.model.layers) * 2 + 1
    if result.attentions is not None:
        assert len(result.attentions) == len(model.model.layers) * 2
    tuple_result = model(input_ids=input_ids, use_cache=False, return_dict=False)
    assert tuple_result[0].shape == (1, 3, 31)
    with pytest.raises(TypeError, match="unsupported forward arguments"):
        model(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))


def test_generation_and_save_reload(tmp_path) -> None:
    model = tiny_model().eval()
    model.tie_weights()
    if model.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    input_ids = torch.tensor([[3, 4]])
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        eos_token_id=None,
        pad_token_id=0,
    )
    assert generated.shape == (1, 4)
    with pytest.raises(ValueError, match="cache_implementation='dynamic' is not supported"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            cache_implementation="dynamic",
            eos_token_id=None,
            pad_token_id=0,
        )
    model.save_pretrained(tmp_path, safe_serialization=False)
    reloaded = RecursiveLlamaForCausalLM.from_pretrained(tmp_path).eval()
    assert torch.allclose(
        model(input_ids=input_ids, use_cache=False).logits,
        reloaded(input_ids=input_ids, use_cache=False).logits,
        atol=1e-6,
        rtol=1e-5,
    )
    register_auto_class()
    auto_reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(auto_reloaded, RecursiveLlamaForCausalLM)
    assert auto_reloaded.config.num_hidden_layers == 4
    assert auto_reloaded.config.recursive_layer_count == 2
    # Transformers 4.54.1 requires the no-config lazy constructor here:
    # config construction forwards unsupported max_cache_len to its cache
    # layer class.  The cache must grow to all logical slots during generation.
    precreated_cache = make_dynamic_cache()
    assert len(precreated_cache) == 0
    generated_with_precreated = auto_reloaded.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        past_key_values=precreated_cache,
        eos_token_id=None,
        pad_token_id=0,
    )
    assert generated_with_precreated.shape == (1, 3)
    assert len(precreated_cache) >= auto_reloaded.config.num_hidden_layers
    assert all(precreated_cache.get_seq_length(index) == input_ids.shape[1] for index in range(4))
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["num_hidden_layers"] == 4
    assert config["recursive_layer_count"] == 2
    assert config["recursive_loops"] == 2
    assert config["recursive_source_layer_indices_0based"] == [0, 3]
