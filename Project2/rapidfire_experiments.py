#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

from sft_data_builder import build_dataset, load_json, save_json


def variant_key(entry):
    return (entry["schema_mode"], entry["output_style"], entry["target_mode"])


def run_group_key(entry):
    return (
        entry["schema_mode"],
        entry["output_style"],
        entry["target_mode"],
        entry.get("model_name", ""),
        entry.get("train_mode", "lora"),
        json.dumps(entry.get("model_kwargs", {}), sort_keys=True),
    )


def variant_filename(split_name, key):
    schema_mode, output_style, target_mode = key
    return f"sft_{split_name}_{schema_mode}_{output_style}_{target_mode}.json"


def ensure_variants(entries, args):
    variants_dir = Path(args.sft_variants_dir)
    variants_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({variant_key(entry) for entry in entries})
    if not args.auto_generate_variants:
        for key in keys:
            for split_name in ("train", "validation"):
                path = variants_dir / variant_filename(split_name, key)
                if not path.exists():
                    raise FileNotFoundError(f"Missing variant file: {path}")
        return
    train_rows = load_json(args.train_json)
    val_rows = load_json(args.validation_json)
    for key in keys:
        for split_name, rows in (("train", train_rows), ("validation", val_rows)):
            out_path = variants_dir / variant_filename(split_name, key)
            payload = build_dataset(
                rows=rows,
                schemas_dir=args.schemas_dir,
                schema_mode=key[0],
                output_style=key[1],
                target_mode=key[2],
                filtered_max_tables=args.filtered_max_tables,
            )
            save_json(str(out_path), payload)


def normalize_links(links):
    if not isinstance(links, dict):
        return {}
    out = {}
    for table_name, cols in links.items():
        table_key = str(table_name).strip()
        if not table_key:
            continue
        col_values = cols if isinstance(cols, list) else []
        clean = sorted({str(col).strip() for col in col_values if str(col).strip()})
        out[table_key] = clean
    return {k: out[k] for k in sorted(out)}


def extract_links(text):
    if isinstance(text, dict):
        if "schema_links" in text:
            return normalize_links(text["schema_links"])
        return normalize_links(text)
    if not isinstance(text, str):
        return None
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right < left:
        return None
    snippet = text[left : right + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "schema_links" in obj:
        return normalize_links(obj["schema_links"])
    return normalize_links(obj)


def table_set(links):
    return set(links.keys())


def pair_set(links):
    out = set()
    for table_name, cols in links.items():
        for col in cols:
            out.add((table_name, col))
    return out


def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not gold_set else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def schema_link_metrics(eval_preds):
    if not isinstance(eval_preds, (tuple, list)) or len(eval_preds) < 2:
        return {}
    predictions, labels = eval_preds
    n = min(len(predictions), len(labels))
    if n == 0:
        return {}
    valid_json = 0
    exact_match = 0
    sum_pt = sum_rt = sum_ft = 0.0
    sum_pc = sum_rc = sum_fc = 0.0
    for pred_text, gold_text in zip(predictions[:n], labels[:n]):
        pred_links = extract_links(pred_text)
        gold_links = extract_links(gold_text)
        if gold_links is None:
            gold_links = {}
        if pred_links is not None:
            valid_json += 1
        else:
            pred_links = {}
        if pred_links == gold_links:
            exact_match += 1
        pt, rt, ft = prf(table_set(pred_links), table_set(gold_links))
        pc, rc, fc = prf(pair_set(pred_links), pair_set(gold_links))
        sum_pt += pt
        sum_rt += rt
        sum_ft += ft
        sum_pc += pc
        sum_rc += rc
        sum_fc += fc
    avg_pt = sum_pt / n
    avg_rt = sum_rt / n
    avg_ft = sum_ft / n
    avg_pc = sum_pc / n
    avg_rc = sum_rc / n
    avg_fc = sum_fc / n
    table_score = (avg_pt + avg_rt + avg_ft) / 3.0
    column_score = (avg_pc + avg_rc + avg_fc) / 3.0
    leaderboard_proxy = 0.5 * table_score + 0.5 * column_score
    return {
        "json_valid_rate": valid_json / n,
        "exact_match_rate": exact_match / n,
        "table_f1_proxy": avg_ft,
        "column_f1_proxy": avg_fc,
        "leaderboard_proxy": leaderboard_proxy,
    }


def row_formatting(row):
    return {"prompt": row["prompt"], "completion": row["completion"]}


def create_model(model_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_kwargs = dict(model_config["model_kwargs"])
    # Some newer architectures (e.g. Qwen3.5) reject use_cache in __init__;
    # apply via config after load instead.
    use_cache = model_kwargs.pop("use_cache", None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if use_cache is not None and hasattr(model, "config"):
        model.config.use_cache = use_cache
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer)


def build_rf_model_config(entry):
    from rapidfireai.automl import RFModelConfig, RFLoraConfig, RFSFTConfig

    train_mode = entry.get("train_mode", "lora")
    if train_mode not in {"lora", "qlora", "full_ft"}:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    peft_config = None
    if train_mode in {"lora", "qlora"}:
        if "lora" not in entry:
            raise ValueError(f"Config {entry.get('config_id')} missing 'lora' block for {train_mode}")
        peft_config = RFLoraConfig(
            r=entry["lora"]["r"],
            lora_alpha=entry["lora"]["alpha"],
            lora_dropout=entry["lora"]["dropout"],
            target_modules=entry["lora"]["target_modules"],
            bias="none",
        )

    model_kwargs = {"device_map": "auto", "torch_dtype": "auto", "use_cache": False}
    model_kwargs.update(entry.get("model_kwargs", {}))
    requested_save_strategy = entry["train"].get("save_strategy", "epoch")
    hf_valid_save_strategies = {"no", "steps", "epoch", "best"}
    hf_save_strategy = (
        requested_save_strategy
        if requested_save_strategy in hf_valid_save_strategies
        else "epoch"
    )
    train_kwargs = {
        "output_dir": entry["artifact_dir"],
        "learning_rate": entry["train"]["learning_rate"],
        "lr_scheduler_type": entry["train"]["lr_scheduler_type"],
        "per_device_train_batch_size": entry["train"]["per_device_train_batch_size"],
        "per_device_eval_batch_size": entry["train"]["per_device_eval_batch_size"],
        "gradient_accumulation_steps": entry["train"]["gradient_accumulation_steps"],
        "max_steps": entry["train"]["max_steps"],
        "logging_steps": entry["train"]["logging_steps"],
        # RFSFTConfig validates HF save strategies only.
        "save_strategy": hf_save_strategy,
        "save_total_limit": entry["train"].get("save_total_limit", 3),
        "bf16": entry["train"]["bf16"],
        "eval_strategy": entry["train"].get("eval_strategy", "steps"),
    }
    if "eval_steps" in entry["train"]:
        train_kwargs["eval_steps"] = entry["train"]["eval_steps"]
    if "save_steps" in entry["train"]:
        train_kwargs["save_steps"] = entry["train"]["save_steps"]
    elif "eval_steps" in entry["train"]:
        train_kwargs["save_steps"] = entry["train"]["eval_steps"]
    # Optional passthrough knobs used in some stronger configs.
    for opt_key in ("max_length", "weight_decay", "max_grad_norm", "gradient_checkpointing", "warmup_ratio"):
        if opt_key in entry["train"]:
            train_kwargs[opt_key] = entry["train"][opt_key]
    train = RFSFTConfig(**train_kwargs)
    if requested_save_strategy == "chunk":
        # RapidFire's worker-level disk persistence checks for the literal
        # "chunk" strategy in config_leaf when shared memory is enabled.
        # Use normal attribute assignment so RapidFire's RF wrapper updates
        # _user_params; mutating __dict__ bypasses config expansion state.
        # Trainer-side HF config is still forced to save_strategy="no"
        # internally by RapidFire, so this marker is safe.
        train.save_strategy = "chunk"
    return RFModelConfig(
        model_name=entry["model_name"],
        peft_config=peft_config,
        training_args=train,
        model_type="causal_lm",
        model_kwargs=model_kwargs,
        formatting_func=row_formatting,
        compute_metrics=schema_link_metrics,
        generation_config=entry["generation"],
    )


def load_hf_dataset(path):
    from datasets import Dataset

    rows = load_json(path)
    return Dataset.from_list(rows)


def run_experiment(entries, args):
    from rapidfireai import Experiment
    from rapidfireai.automl import List, RFGridSearch

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")
    grouped = defaultdict(list)
    for entry in entries:
        grouped[run_group_key(entry)].append(entry)
    for key, group_entries in grouped.items():
        dataset_key = key[:3]
        train_path = Path(args.sft_variants_dir) / variant_filename("train", dataset_key)
        eval_path = Path(args.sft_variants_dir) / variant_filename("validation", dataset_key)
        train_ds = load_hf_dataset(str(train_path))
        eval_ds = load_hf_dataset(str(eval_path))
        prepared_entries = []
        artifact_root = Path(args.artifact_root).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        for entry in group_entries:
            copied = dict(entry)
            copied["artifact_dir"] = str((artifact_root / entry["config_id"]).resolve())
            Path(copied["artifact_dir"]).mkdir(parents=True, exist_ok=True)
            prepared_entries.append(copied)
        rf_configs = [build_rf_model_config(entry) for entry in prepared_entries]
        config_group = RFGridSearch(configs=List(rf_configs), trainer_type="SFT")
        experiment.run_fit(
            config_group,
            create_model,
            train_ds,
            eval_ds,
            num_chunks=args.num_chunks,
            seed=args.seed,
        )
    experiment.end()


def print_plan(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[variant_key(entry)].append(entry["config_id"])
    print("Config groups by dataset variant:")
    for key in sorted(grouped.keys()):
        print(f"- {key}: {grouped[key]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_matrix_json", default="configs/config_matrix.json")
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--validation_json", default="validation.json")
    parser.add_argument("--sft_variants_dir", default="sft_variants")
    parser.add_argument("--artifact_root", default="adapter")
    parser.add_argument("--experiment_name", default="project2-schema-linking")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filtered_max_tables", type=int, default=8)
    parser.add_argument("--auto_generate_variants", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_json(args.config_matrix_json)
    if len(entries) < 1:
        raise ValueError("Config matrix must contain at least 1 config.")
    ensure_variants(entries, args)
    print_plan(entries)
    if args.dry_run:
        print("Dry run complete. No RapidFire training launched.")
        return
    run_experiment(entries, args)


if __name__ == "__main__":
    main()
