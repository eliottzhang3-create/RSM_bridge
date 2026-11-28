"""Official native Mizar; complete S3 pool, exact half-epoch fine-tuning budget."""
from __future__ import annotations
import hashlib,json,math,os,random,time
from pathlib import Path
ROUTE = "mizar_final_s3_full_projector_lm_halfepoch_gbs128_v1"
WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, EPOCHS, SEED = 8,4,4,1,1234
TRAIN_FRACTION = .5
MAX_LR, MIN_LR, WARMUP_RATIO, KEEP_CHECKPOINTS = 5e-6,1e-6,.05,4
RANK_UPDATE_SAMPLES = BATCH_SIZE * ACCUMULATION_STEPS
GLOBAL_BATCH_SIZE = WORLD_SIZE * RANK_UPDATE_SAMPLES
PROMPT_TOKENS, ANSWER_TOKENS, AUDIO_TOKENS = 256,96,126
PREFIX_TOKENS = 383
PAD_TOKEN, SEPARATOR_TOKEN_ID, EOS_TOKEN_ID = "!",0,0
SOURCE_COUNTS = {"AVQA":26481,"AudioCaps":49486,"Clotho":1991,"CompA-R":48045,
                 "LP-MusicCaps-MTT":7871,"MusicCaps":631,"SpeechCraft":113975,"TACOS":24316}
TOTAL_ROWS = sum(SOURCE_COUNTS.values())
DEFAULT_DATA_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1"
DEFAULT_SOURCE = DEFAULT_DATA_DIR + "/train.json"
DEFAULT_RESOLUTIONS = DEFAULT_DATA_DIR + "/resolved_paths.tsv"
SCHEDULE = tuple(range(30))
CORE_LAYERS = tuple(range(30))
DEFAULT_MODEL_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M"
CED_ASSET_FILES = ("config.json","configuration_ced.py","modeling_ced.py","feature_extraction_ced.py","preprocessor_config.json","model.safetensors")
CED_EXPECTED_CONFIG = {"embed_dim":384,"depth":12,"num_heads":6,"n_mels":64,"patch_size":16,"patch_stride":16,"target_length":1012,"n_fft":512,"win_size":512,"hop_size":160,"center":True,"f_min":0,"f_max":8000,"pooling":"mean","pad_last":True,"outputdim":527}


def ced_label_metadata(config):
    """Read the real local labels, retaining duplicate display names by index."""
    raw = config.get("id2label")
    error = "incompatible CED-Small assets: config.json needs complete local id2label metadata (indices 0..526)"
    if not isinstance(raw, dict) or len(raw) != 527:
        raise ValueError(error)
    labels = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError(error) from None
        if str(index) != str(key) or not isinstance(value, str) or not value.strip():
            raise ValueError(error)
        labels[index] = value
    if set(labels) != set(range(527)):
        raise ValueError(error)
    labels = dict(sorted(labels.items()))
    reverse = config.get("label2id")
    if reverse is None:
        reverse = {label: index for index, label in labels.items()}
    if (not isinstance(reverse, dict) or set(reverse) != set(labels.values())
            or any(type(index) is not int or index not in labels or labels[index] != label
                   for label, index in reverse.items())):
        raise ValueError("incompatible CED-Small assets: inconsistent local label2id metadata")
    return labels, dict(reverse)

def validate_ced_config(config):
    """Reject incompatible feature geometry instead of reshaping/truncating it."""
    if not isinstance(config, dict):
        raise ValueError("incompatible CED-Small assets: config.json must contain an object")
    problems = [f"{name}={config.get(name)!r}, expected {expected!r}"
                for name, expected in CED_EXPECTED_CONFIG.items()
                if config.get(name) != expected]
    if config.get("model_type") != "ced":
        problems.append(f"model_type={config.get('model_type')!r}, expected 'ced'")
    auto_map = config.get("auto_map", {})
    if not isinstance(auto_map, dict):
        raise ValueError("incompatible CED-Small assets: auto_map must contain an object")
    for name, expected in (("AutoConfig", "configuration_ced.CedConfig"),
                           ("AutoModelForAudioClassification", "modeling_ced.CedForAudioClassification")):
        if auto_map.get(name) != expected:
            problems.append(f"auto_map.{name}={auto_map.get(name)!r}, expected {expected!r}")
    if problems:
        raise ValueError("incompatible CED-Small assets: " + "; ".join(problems))
    labels, _ = ced_label_metadata(config)
    return {**CED_EXPECTED_CONFIG, "input_sampling_rate": 32000, "ced_sampling_rate": 16000,
            "mel_frames": 2001, "segments": 2, "hidden_tokens_per_segment": 252,
            "hidden_tokens": 504, "temporal_tokens": AUDIO_TOKENS, "output_dim": 576,
            "mapper": "freq_merge_126", "mapper_dropout": 0.05, "mapper_resample": False,
            "label_metadata": "local_config_json", "label_count": len(labels)}

def validate_ced_assets(root):
    root = Path(root).resolve()
    for name in CED_ASSET_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty local CED asset: {path}")
    return validate_ced_config(json.loads((root / "config.json").read_text(encoding="utf-8")))

def validate_tokenizer_contract(tokenizer, pad_id, embedding_rows):
    """Validate the Mizar text-token contract without assuming PAD aliases EOS.

    Mizar registers the existing token ``!`` as padding.  In the checked
    SmolLM2 vocabulary that token is ID 17; ID 0 remains both EOS and the
    single audio/text separator.  Derive the padding ID from the tokenizer so
    this remains valid for an unchanged compatible tokenizer, while requiring
    it to be distinct from the separator/EOS token.
    """
    tokenizer_rows = len(tokenizer)
    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # This intentionally mirrors AudioTextDataset and the released Mizar
    # trainer, which derive ignore_index with encode(tokenizer.pad_token)[0].
    encoded_pad = tokenizer.encode(tokenizer.pad_token)
    audit = {
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_id": int(pad_id),
        "tokenizer_pad_id": tokenizer_pad_id,
        "sep_id": SEPARATOR_TOKEN_ID,
        "eos_id": eos_id,
        "encoded_pad_ids": encoded_pad,
        "tokenizer_vocab_rows": tokenizer_rows,
        "embedding_rows": int(embedding_rows),
    }
    problems = []
    if audit["pad_token"] != PAD_TOKEN:
        problems.append(f"pad_token={audit['pad_token']!r}, expected {PAD_TOKEN!r}")
    if audit["pad_id"] != tokenizer_pad_id:
        problems.append(f"dataset pad_id={audit['pad_id']}, tokenizer.pad_token_id={tokenizer_pad_id}")
    if not encoded_pad or encoded_pad[0] != audit["pad_id"]:
        problems.append(f"encode({PAD_TOKEN!r})={encoded_pad}, expected first ID {audit['pad_id']}")
    if audit["pad_id"] in {SEPARATOR_TOKEN_ID, EOS_TOKEN_ID}:
        problems.append(f"pad_id={audit['pad_id']} aliases SEP/EOS={SEPARATOR_TOKEN_ID}")
    if eos_id != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={eos_id}, expected {EOS_TOKEN_ID}")
    if tokenizer_rows != embedding_rows:
        problems.append(f"tokenizer rows={tokenizer_rows}, embedding rows={embedding_rows}")
    if not 0 <= audit["pad_id"] < embedding_rows:
        problems.append(f"pad_id={audit['pad_id']} is outside embedding rows={embedding_rows}")
    if problems:
        details = ", ".join(f"{key}={value!r}" for key, value in audit.items())
        raise ValueError("invalid Mizar tokenizer contract: " + "; ".join(problems) + "; " + details)
    return audit

def validate_checkpoint_tokenizer(tokenizer, model_vocab_size):
    """Apply and audit Mizar's exact tokenizer padding operation.

    Recursive checkpoints commonly serialize EOS as their source padding
    token.  Mizar unconditionally calls ``add_special_tokens`` with the
    existing ``!`` vocabulary token, so the source PAD identity is
    informational.  The important constraints are the post-override
    ``!``/PAD contract, EOS=0, and no vocabulary growth.
    """
    try:
        model_vocab_size = int(model_vocab_size)
    except (TypeError, ValueError):
        model_vocab_size = -1
    source_rows = len(tokenizer)
    source_pad_id = getattr(tokenizer, "pad_token_id", None)
    source_pad_token = getattr(tokenizer, "pad_token", None)
    source_encoded_pad = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)
    added_tokens = tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer_rows = len(tokenizer)
    # Fast tokenizers can return 1 when registering an existing vocabulary
    # token as special, while keeping both its ID and vocabulary size intact.
    # The API return value is diagnostic, not a vocabulary-size delta.
    vocabulary_growth = tokenizer_rows - source_rows
    # Keep this call identical to AudioTextDataset/Mizar rather than relying
    # on a separately derived token ID.  The no-special-token encoding below
    # separately proves that ``!`` is one existing vocabulary token.
    pad_encoding = tokenizer.encode(tokenizer.pad_token)
    if not pad_encoding:
        raise ValueError("Mizar pad token encoding is empty")
    pad_id = pad_encoding[0]
    encoded_pad = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    audit = {
        "source": "recursive_checkpoint",
        "source_pad_token": source_pad_token,
        "source_pad_id": source_pad_id,
        "source_mizar_pad_ids": source_encoded_pad,
        "mizar_pad_token": PAD_TOKEN,
        "mizar_pad_id": pad_id,
        "mizar_pad_ids": encoded_pad,
        "mizar_pad_encoding": pad_encoding,
        "added_tokens": int(added_tokens),
        "vocabulary_growth": vocabulary_growth,
        "eos_id": eos_id,
        "source_vocab_rows": source_rows,
        "tokenizer_vocab_rows": tokenizer_rows,
        "model_vocab_size": model_vocab_size,
        "existing_pad_contract": (
            vocabulary_growth == 0 and source_encoded_pad == encoded_pad == [pad_id]
        ),
    }
    problems = []
    if eos_id != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={eos_id!r}, expected {EOS_TOKEN_ID}")
    if model_vocab_size <= 0 or tokenizer_rows != model_vocab_size:
        problems.append(
            f"tokenizer rows={tokenizer_rows}, model vocab_size={model_vocab_size}"
        )
    if vocabulary_growth != 0:
        problems.append(
            f"Mizar pad assignment changed vocabulary rows from {source_rows} to {tokenizer_rows}; "
            f"expected existing {PAD_TOKEN!r} (add_special_tokens returned {added_tokens})"
        )
    if source_encoded_pad != encoded_pad:
        problems.append(
            f"Mizar pad assignment changed {PAD_TOKEN!r} token IDs from "
            f"{source_encoded_pad!r} to {encoded_pad!r}"
        )
    if tokenizer.pad_token != PAD_TOKEN:
        problems.append(f"pad_token={tokenizer.pad_token!r}, expected {PAD_TOKEN!r}")
    if len(encoded_pad) != 1 or encoded_pad != [pad_id]:
        problems.append(
            f"encode({PAD_TOKEN!r})={encoded_pad!r}, expected [{pad_id}]"
        )
    elif not 0 <= int(pad_id) < min(tokenizer_rows, model_vocab_size):
        problems.append(f"{PAD_TOKEN!r} token ID {pad_id} is outside the model vocabulary")
    elif int(pad_id) == EOS_TOKEN_ID:
        problems.append(f"{PAD_TOKEN!r} token ID aliases EOS/SEP={EOS_TOKEN_ID}")
    if source_pad_id is not None and not 0 <= int(source_pad_id) < tokenizer_rows:
        problems.append(f"source pad_token_id={source_pad_id!r} is outside the tokenizer vocabulary")
    if problems:
        raise ValueError("invalid recursive checkpoint tokenizer: " + "; ".join(problems))
    return audit

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers briefly deny replacing an open file. Linux
            # renames remain atomic immediately; never hide a real Unix error.
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(.01)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()

def code_digest():
    root=Path(__file__).resolve().parent
    files={path.relative_to(root).as_posix():path for path in root.rglob("*")
           if path.suffix in {".py",".sh"} and "__pycache__" not in path.parts}
    files["release_manifest.json"]=root/"release_manifest.json"
    files["../run_mizar_s3_full_finetune_3090.sh"]=root.parent/"run_mizar_s3_full_finetune_3090.sh"
    for path in (root.parent/"mizar_s3_data_prepare").glob("*.py"):
        files["../mizar_s3_data_prepare/"+path.name]=path
    files["../prepare_mizar_s3_data.sh"]=root.parent/"prepare_mizar_s3_data.sh"
    result=hashlib.sha256()
    for name,path in sorted(files.items()):
        result.update(name.encode("utf-8")+b"\0")
        result.update(path.read_bytes().replace(b"\r\n",b"\n"))
    return result.hexdigest()


def data_order(rows):
    # The sampler applies the sole global permutation to the complete pool.
    return list(range(rows))


def rank_seed(rank):
    return SEED + rank


def raw_references(row):
    if row.get("source") not in {"audiomcq","avqa"} or row.get("filepath2") != "":
        raise ValueError("S3 requires a verified original single recording")
    if not isinstance(row.get("filepath1"),str) or not row["filepath1"]:
        raise ValueError("S3 lacks original filepath1")
    return row["source_dataset"], [("filepath1",row["filepath1"])]


def validate_data_counts(rows, sources, views=None):
    if type(rows) is not int or rows != TOTAL_ROWS or sources != SOURCE_COUNTS:
        raise ValueError("requires the user-confirmed 272796-row filtered S3 pool and its exact source counts")
    return rows


def target_rows(rows):
    if type(rows) is not int or rows < 2:
        raise ValueError("half epoch requires at least two pool rows")
    return rows // 2


def rank_shard(rows, rank, world_size=WORLD_SIZE):
    # rows is the FULL pool size; shard the exact half-epoch budget.
    base, extra = divmod(target_rows(rows), world_size)
    return rank*base+min(rank,extra), base+int(rank<extra)


def rank_microbatch_size(rows, rank, update, micro):
    count = rank_shard(rows,rank)[1]
    return max(0,min(BATCH_SIZE,count-update*RANK_UPDATE_SAMPLES-micro*BATCH_SIZE))


def geometry(rows):
    budget=target_rows(rows)
    steps=math.ceil(budget/GLOBAL_BATCH_SIZE)
    return {"rows":rows,"target_rows":budget,"world_size":WORLD_SIZE,"microbatch":BATCH_SIZE,
            "gradient_accumulation_steps":ACCUMULATION_STEPS,"global_batch_size":GLOBAL_BATCH_SIZE,
            "epochs":TRAIN_FRACTION,"actual_epoch_fraction":budget/rows,
            "rank_samples_per_optimizer_step":RANK_UPDATE_SAMPLES,
            "rank_samples":[rank_shard(rows,r)[1] for r in range(WORLD_SIZE)],
            "steps_per_epoch":steps,"total_steps":steps,
            "warmup_steps":min(round(steps*WARMUP_RATIO),steps-1),
            "full_updates":budget//GLOBAL_BATCH_SIZE,
            "last_update_rows":budget%GLOBAL_BATCH_SIZE or GLOBAL_BATCH_SIZE,
            "dropped_budget_rows":0,"duplicated_budget_rows":0,
            "unconsumed_pool_rows":rows-budget}


def accumulation_for_update(update, rows=TOTAL_ROWS):
    remaining=target_rows(rows)-update*GLOBAL_BATCH_SIZE
    if remaining<=0:raise ValueError("update outside the half-epoch budget")
    return min(ACCUMULATION_STEPS,math.ceil(remaining/(WORLD_SIZE*BATCH_SIZE)))


def lr_factor(step, total_steps):
    warmup=min(round(total_steps*WARMUP_RATIO),total_steps-1)
    if warmup and step<warmup:return max(.1,(step+1)/warmup)
    progress=min(1.,max(0.,(step-warmup+1)/max(1,total_steps-warmup)))
    return MIN_LR/MAX_LR+(1.-MIN_LR/MAX_LR)*.5*(1.+math.cos(math.pi*progress))


def parameter_role(name):
    if name.startswith("audio_encoder.base.ced."):return "frozen"
    if name.startswith("audio_encoder.base.mapper."):return "mapper"
    if name.startswith("caption_decoder.lm."):return "lm"
    return "frozen"


def checkpoint_steps(rows):
    steps=geometry(rows)["total_steps"]
    return sorted({math.ceil(q*steps/4) for q in range(1,5)})


def training_cursor(step,g):
    if type(step) is not int or not 0<=step<=g["total_steps"]:
        raise ValueError("invalid half-epoch optimizer cursor")
    return 0,step


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def semantic_contract(prepared, model_dir, bundle):
    return {"route":ROUTE,"code_sha256":code_digest(),"geometry":geometry(prepared["rows"]),
            "rows_canonical_sha256":prepared["rows_canonical_sha256"],
            "mapping_sha256":prepared["mapping_sha256"],"source_rows":prepared["source_rows"],
            "dataset_selection":prepared["dataset_selection"],"model_dir":str(model_dir),
            "official_bundle":bundle,"physical_layers":30,"schedule":list(SCHEDULE),
            "trainable_layers_1based":list(range(1,31)),"recurrence_passes":1,"added_parameters":0,
            "projector_trainable":True,"full_lm_trainable":True,"ced_frozen_eval":True,
            "prefix":[126,1,256],"answer_tokens":ANSWER_TOKENS,"sequence_tokens":479,
            "prompt_eos":False,"answer_eos":True,"attention_mask":None,"pad_id":17,"eos_sep_id":0,
            "audio_policy":"resample32k_then_mono_first20_right_pad640000_whole_CED",
            "audio_read_policy":"shared_until_complete_collective_optimizer_boundary_switch",
            "forward_dtype":"float32","autocast":False,"tf32":False,"seed":SEED,
            "epochs":TRAIN_FRACTION,"pool_selection":"full_pool_global_permutation_prefix_half_budget",
            "adamw":{"max_lr":MAX_LR,"min_lr":MIN_LR,"betas":[.9,.999],"eps":1e-8,
                      "weight_decay":.01,"no_decay_bias_ndim_le1":True},
            "scheduler":"5pct_round_warmup_then_cosine_final_real_update_at_min",
            "checkpoint_steps":checkpoint_steps(prepared["rows"]),"keep_checkpoints":KEEP_CHECKPOINTS,
            "loss":"rank_local_microbatch_valid_answer_token_mean_then_actual_GA_then_active_rank_mean",
            "full_bptt":True,"use_cache":False,"gradient_checkpointing":False}
