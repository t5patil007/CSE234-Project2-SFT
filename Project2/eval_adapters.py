#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_data_builder import build_prompt, load_schema, serialize_schema


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def normalize_links(raw_links, schema_info):
    if not isinstance(raw_links, dict):
        return {}
    normalized = {}
    lc_tables = schema_info["table_by_lc"]
    col_by_lc = schema_info["col_by_lc_table"]
    for table_name, cols in raw_links.items():
        canonical_table = lc_tables.get(str(table_name).lower())
        if not canonical_table:
            continue
        col_values = cols if isinstance(cols, list) else []
        seen = set()
        clean_cols = []
        for col_name in col_values:
            canonical_col = col_by_lc.get(canonical_table, {}).get(str(col_name).lower())
            if canonical_col and canonical_col not in seen:
                seen.add(canonical_col)
                clean_cols.append(canonical_col)
        normalized[canonical_table] = sorted(clean_cols)
    return {t: normalized[t] for t in sorted(normalized)}


def extract_links(text):
    if isinstance(text, dict):
        if "schema_links" in text:
            return text["schema_links"]
        return text
    if not isinstance(text, str):
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right < left:
        return {}
    snippet = text[left : right + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict) and "schema_links" in payload:
        return payload["schema_links"]
    return payload if isinstance(payload, dict) else {}


def generate_for_adapter(
    adapter_dir,
    questions,
    schemas_dir,
    schema_mode,
    filtered_max_tables,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    limit=None,
):
    adapter_dir = Path(adapter_dir).resolve()
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    adapter_cfg = load_json(adapter_cfg_path)
    base_model = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    do_sample = temperature > 0.0
    selected = questions[:limit] if limit else questions

    schema_cache = {}
    preds = []
    for idx, item in enumerate(selected, start=1):
        db_id = item["db_id"]
        schema_info = schema_cache.get(db_id)
        if schema_info is None:
            schema_info = load_schema(db_id, schemas_dir)
            schema_cache[db_id] = schema_info

        schema_text = serialize_schema(
            schema=schema_info,
            question=item["question"],
            schema_mode=schema_mode,
            filtered_max_tables=filtered_max_tables,
        )
        prompt_messages = build_prompt(item["question"], db_id, schema_text)
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_links = extract_links(raw_text)
        links = normalize_links(raw_links, schema_info)
        preds.append({"question_id": item["question_id"], "schema_links": links})

        if idx % 25 == 0:
            print(f"[{adapter_dir.name}] generated {idx}/{len(selected)} predictions")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, base_model


def run_eval(eval_py, predictions, gold, schemas_dir, questions_input, per_question_out):
    cmd = [
        sys.executable,
        str(eval_py),
        "--predictions",
        str(predictions),
        "--gold",
        str(gold),
        "--schemas_dir",
        str(schemas_dir),
        "--questions_input",
        str(questions_input),
        "--per_question_out",
        str(per_question_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"eval.py failed for {predictions}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    text = result.stdout
    score_match = re.search(r"Leaderboard Score\s*:\s*([0-9.]+)", text)
    score = float(score_match.group(1)) if score_match else None
    return score, text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dirs", nargs="+", required=True)
    parser.add_argument("--questions_input", default="validation_input.json")
    parser.add_argument("--gold", default="validation_gold_schema_links.json")
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--eval_py", default="eval.py")
    parser.add_argument("--schema_mode", default="minimal", choices=["minimal", "with_types", "with_pkfk", "filtered"])
    parser.add_argument("--filtered_max_tables", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preds_out_dir", default="/root/logs/adapter_eval/predictions")
    parser.add_argument("--reports_out_dir", default="/root/logs/adapter_eval/reports")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    questions_input = (project_root / args.questions_input).resolve()
    gold = (project_root / args.gold).resolve()
    schemas_dir = (project_root / args.schemas_dir).resolve()
    eval_py = (project_root / args.eval_py).resolve()
    preds_out_dir = (project_root / args.preds_out_dir).resolve()
    reports_out_dir = (project_root / args.reports_out_dir).resolve()
    preds_out_dir.mkdir(parents=True, exist_ok=True)
    reports_out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_json(questions_input)
    summary = []

    for adapter in args.adapter_dirs:
        adapter_dir = Path(adapter).resolve()
        label = adapter_dir.parent.name + "_" + adapter_dir.parent.parent.name + "_" + adapter_dir.parent.parent.parent.name if adapter_dir.name == "final_checkpoint" else adapter_dir.name
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)

        print(f"Evaluating adapter: {adapter_dir}")
        preds, base_model = generate_for_adapter(
            adapter_dir=adapter_dir,
            questions=questions,
            schemas_dir=schemas_dir,
            schema_mode=args.schema_mode,
            filtered_max_tables=args.filtered_max_tables,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            limit=args.limit,
        )

        pred_path = preds_out_dir / f"{label}.json"
        per_q_path = reports_out_dir / f"{label}_per_question.csv"
        eval_stdout_path = reports_out_dir / f"{label}_eval.txt"
        save_json(pred_path, preds)

        score, eval_stdout = run_eval(
            eval_py=eval_py,
            predictions=pred_path,
            gold=gold,
            schemas_dir=schemas_dir,
            questions_input=questions_input,
            per_question_out=per_q_path,
        )
        eval_stdout_path.write_text(eval_stdout)
        print(eval_stdout)

        summary.append(
            {
                "label": label,
                "adapter_dir": str(adapter_dir),
                "base_model": base_model,
                "schema_mode": args.schema_mode,
                "predictions_path": str(pred_path),
                "per_question_csv": str(per_q_path),
                "eval_stdout_path": str(eval_stdout_path),
                "leaderboard_score": score,
            }
        )

    summary.sort(
        key=lambda row: (-1.0 if row["leaderboard_score"] is None else -row["leaderboard_score"])
    )
    summary_json = reports_out_dir / "summary.json"
    summary_csv = reports_out_dir / "summary.csv"
    save_json(summary_json, summary)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n=== Ranked results ===")
    for idx, row in enumerate(summary, start=1):
        print(f"{idx}. {row['label']} -> leaderboard_score={row['leaderboard_score']}")
    print(f"\nWrote: {summary_json}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
