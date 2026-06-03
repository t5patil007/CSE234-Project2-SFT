#!/usr/bin/env python3
"""
main.py -- Schema-linking inference for Project 2 submission.

CLI (required):
    python3 main.py --input <questions.json> --output <predictions.json>

Optional:
    --schemas_dir ./schemas   (default; resolved from current working directory)
    --checkpoint_path <dir>   (LoRA adapter directory with adapter_config.json)

Input JSON list: {question_id, db_id, question}
Output JSON list: {question_id, schema_links}  (no db_id; canonical table/column casing)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_adapters import extract_links, normalize_links
from sft_data_builder import build_prompt, load_schema, serialize_schema

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = (
    SCRIPT_DIR
    / "adapter/w2_cfg07_lora_qwen15b_minimal_lr2e5_a100_ctrl_noaug/final_checkpoint"
)

SCHEMA_MODE = "minimal"
FILTERED_MAX_TABLES = 8
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.05
TOP_P = 0.9
TOP_K = 30
REPETITION_PENALTY = 1.02

_model = None
_tokenizer = None
_schema_cache = {}


def resolve_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    else:
        path = path.resolve()
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Checkpoint not found or missing adapter_config.json: {path}"
        )
    return path


def load_model(checkpoint_dir: Path):
    global _model, _tokenizer
    if _model is not None:
        return

    with open(checkpoint_dir / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()

    _model = model
    _tokenizer = tokenizer


def sanitize_schema_links(links):
    """Ensure schema_links is a dict[str, list] with list values only."""
    if not isinstance(links, dict):
        return {}
    out = {}
    for table_name, cols in links.items():
        if not isinstance(cols, list):
            cols = []
        out[str(table_name)] = [str(c) for c in cols]
    return out


def predict_schema_links(question, db_id, schemas_dir):
    """Predict schema_links for one (question, db_id) pair."""
    schema_info = _schema_cache.get(db_id)
    if schema_info is None:
        schema_info = load_schema(db_id, schemas_dir)
        _schema_cache[db_id] = schema_info

    schema_text = serialize_schema(
        schema=schema_info,
        question=question,
        schema_mode=SCHEMA_MODE,
        filtered_max_tables=FILTERED_MAX_TABLES,
    )
    prompt_messages = build_prompt(question, db_id, schema_text)
    prompt = _tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = TEMPERATURE > 0.0
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=TEMPERATURE if do_sample else None,
            top_p=TOP_P if do_sample else None,
            top_k=TOP_K if do_sample else None,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_links = extract_links(raw_text)
    return sanitize_schema_links(normalize_links(raw_links, schema_info))


def load_input_items(path: Path):
    with path.open() as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of question objects.")
    required = {"question_id", "db_id", "question"}
    for idx, row in enumerate(items):
        if not isinstance(row, dict):
            raise ValueError(f"Input row {idx} must be a JSON object.")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Input row {idx} missing fields: {sorted(missing)}")
    return items


def main():
    ap = argparse.ArgumentParser(
        description="Generate schema_links predictions for Project 2."
    )
    ap.add_argument("--input", required=True, help="Input questions JSON path")
    ap.add_argument("--output", required=True, help="Output predictions JSON path")
    ap.add_argument(
        "--schemas_dir",
        default="./schemas",
        help="Directory containing released schema JSON files (default: ./schemas from cwd)",
    )
    ap.add_argument(
        "--checkpoint_path",
        default=str(DEFAULT_CHECKPOINT_DIR.relative_to(SCRIPT_DIR)),
        help="LoRA adapter directory (default: bundled adapter under this repo)",
    )
    args = ap.parse_args()

    schemas_path = Path(args.schemas_dir)
    if not schemas_path.is_dir():
        print(f"ERROR: schemas_dir not found: {schemas_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    schemas_dir = str(schemas_path.resolve())

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)
        load_model(checkpoint_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    items = load_input_items(input_path)

    preds = []
    for row in items:
        links = predict_schema_links(
            question=str(row["question"]),
            db_id=str(row["db_id"]),
            schemas_dir=schemas_dir,
        )
        preds.append({"question_id": row["question_id"], "schema_links": links})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {output_path}")


if __name__ == "__main__":
    main()
