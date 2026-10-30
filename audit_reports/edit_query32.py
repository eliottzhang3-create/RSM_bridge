from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn'
DST = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine' / NAME

def edit(rel, old, new, count=None):
    path = DST / rel
    value = path.read_text(encoding='utf-8')
    actual = value.count(old)
    if not actual or count is not None and actual != count:
        raise ValueError((rel, old, actual, count))
    path.write_text(value.replace(old, new), encoding='utf-8', newline='\n')

edit('optim.py', '"audio_input_norms", "audio_output_norms", "text_norm"}',
     '"audio_input_norms", "audio_output_norms", "text_norm",\n'
     '                             "query_tokens", "compression_attention", "query_norm"}')
edit('optim.py', '"crossattn_text_norm"}:',
     '"crossattn_text_norm", "crossattn_query_tokens",\n'
     '                                       "crossattn_compression_attention", "crossattn_query_norm"}:')
edit('train.py', '"crossattn_audio_output_norms", "crossattn_text_norm"}',
     '"crossattn_audio_output_norms", "crossattn_text_norm", "crossattn_query_tokens",\n'
     '                       "crossattn_compression_attention", "crossattn_query_norm"}')
edit('train.py', '"crossattn_audio_output_norms", "crossattn_text_norm")}',
     '"crossattn_audio_output_norms", "crossattn_text_norm", "crossattn_query_tokens",\n'
     '                 "crossattn_compression_attention", "crossattn_query_norm")}')
edit('train.py', '"gate1", "gate2", "cross_attention", "round_bias.0", "round_bias.1", "text_norm",',
     '"gate1", "gate2", "cross_attention", "round_bias.0", "round_bias.1", "text_norm",\n'
     '                 "query_tokens", "compression_attention", "query_norm",')

edit('models/recursive_text.py', 'adapter.prompt_tokens != 256 or adapter.cross_attention.num_heads != 9',
     'adapter.prompt_tokens != 256 or adapter.memory_tokens != 32 or adapter.cross_attention.num_heads != 9')
edit('models/recursive_text.py', '    if type(adapter.residual_scale)',
     '    if (adapter.compression_attention is adapter.cross_attention\n'
     '            or adapter.compression_attention.embed_dim != 576\n'
     '            or adapter.compression_attention.num_heads != 9\n'
     '            or adapter.compression_attention.dropout != 0.\n'
     '            or not adapter.compression_attention.batch_first):\n'
     '        raise ValueError("shared query compression attention geometry/independence mismatch")\n'
     '    if type(adapter.residual_scale)')
edit('models/recursive_text.py', '(adapter.output_norm, 576, False)]',
     '(adapter.output_norm, 576, False),\n'
     '              (adapter.query_norm, 576, True), (adapter.memory_norm, 576, False)]')
edit('models/recursive_text.py', '    expected["text_norm.weight"] = (576,)',
     '    expected.update({"query_tokens": (32, 576),\n'
     '                     "compression_attention.in_proj_weight": (1728, 576),\n'
     '                     "compression_attention.in_proj_bias": (1728,),\n'
     '                     "compression_attention.out_proj.weight": (576, 576),\n'
     '                     "compression_attention.out_proj.bias": (576,),\n'
     '                     "query_norm.weight": (576,), "query_norm.bias": (576,)})\n'
     '    expected["text_norm.weight"] = (576,)')

edit('contract.py', '"new_parameters": 4892741', '"new_parameters": 6241733')
edit('contract.py', '        "logical_indices_0based": [5, 15], "text_start": 127,',
     '        "logical_indices_0based": [5, 15], "text_start": 127,\n'
     '        "query_compression": {"query_tokens": 32, "width": 576, "input_tokens": 126,\n'
     '            "heads": 9, "head_dim": 64, "dropout": 0., "bias": True,\n'
     '            "query_parameter_shape": [32, 576], "query_init_std": 0.02,\n'
     '            "shared_queries_across_depths": True, "shared_attention_across_depths": True,\n'
     '            "independent_of_text_attention": True, "independent_depth_reads": True,\n'
     '            "query_norm": {"eps": 1e-5, "elementwise_affine": True},\n'
     '            "memory_norm": {"eps": 1e-5, "elementwise_affine": False},\n'
     '            "query_residual": False, "self_attention": False, "ffn": False,\n'
     '            "extra_position_encoding": False, "text_conditioned": False,\n'
     '            "bank_shape": [4, 32, 576], "once_per_lm_forward": True,\n'
     '            "slot_semantics": "learned_summary_slots_not_ordered_time_frames",\n'
     '            "query_weight_decay": 0.01},')
edit('write_config.py', '                  "shared text LN feeds',
     '                  "shared32 learned queries + query LN + independent shared MHA depth reads + non-affine memory LN; bank4x32x576",\n'
     '                  "shared text LN feeds')

# Retain the original base audit and additionally prove the immediate v2 parent
# remains equal in every component that query compression does not modify.
edit('audit_parent.py', '    import contract\n    return {',
     '    v2_parent = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn")\n'
     '    v2_equal = unchanged + ["models/ced.py", "models/decoder.py", "models/mellow.py",\n'
     '                            "models/recursive_crossattn.py"]\n'
     '    for name in v2_equal:\n'
     '        if source(v2_parent / name) != source(ROOT / name):\n'
     '            raise AssertionError(f"query32 changed unchanged v2 implementation: {name}")\n'
     '    if dump(function(tree(v2_parent / "train.py"), "train")) != dump(candidate_train):\n'
     '        raise AssertionError("query32 changed the v2 formal training loop")\n'
     '    import contract\n    return {')
edit('audit_parent.py', '            "parent": PARENT.name,',
     '            "query32_parent": v2_parent.name, "v2_identical_files": v2_equal,\n'
     '            "v2_complete_training_function_ast_equal": True,\n'
     '            "parent": PARENT.name,')

print('query32 contracts, trainability, gradients, optimizer, and parity updated')
