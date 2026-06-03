import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_adapters import extract_links, normalize_links
from sft_data_builder import build_prompt, load_schema, serialize_schema

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = (
    PROJECT_ROOT
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


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    adapter_dir = DEFAULT_ADAPTER_DIR.resolve()
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing adapter weights at {adapter_dir}. "
            "Train or copy w2_cfg07_lora_qwen15b_minimal_lr2e5_a100_ctrl_noaug first."
        )

    with open(cfg_path) as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg["base_model_name_or_path"]

    _tokenizer = AutoTokenizer.from_pretrained(base_model)
    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    _model = PeftModel.from_pretrained(_model, str(adapter_dir))
    _model.eval()


def predict_schema_links(question, db_id, schemas_dir):
    """Run fine-tuned Qwen2.5-1.5B ctrl LoRA schema linking for one question."""
    _load_model()

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
    return normalize_links(raw_links, schema_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")
    args = ap.parse_args()

    schemas_dir = str((PROJECT_ROOT / args.schemas_dir).resolve())
    with open(args.input) as f:
        items = json.load(f)

    preds = []
    for it in items:
        links = predict_schema_links(it["question"], it["db_id"], schemas_dir)
        preds.append({"question_id": it["question_id"], "schema_links": links})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
