#!/usr/bin/env python3
"""Build per-strategy config matrices for the final Qwen3.5-2B model swap experiment.

Mirrors the w2_cfg07 rows from the top-3 augmentation matrices (ctrl_noaug and
augB_targeted_hard), swapping Qwen2.5-1.5B-Instruct -> Qwen/Qwen3.5-2B.
"""
import argparse
import copy
import json
from pathlib import Path

BASE_CONFIG_ID = "w2_cfg07_lora_qwen15b_minimal_lr2e5_a100"
QWEN35_MODEL = "Qwen/Qwen3.5-2B"

STRATEGIES = {
    "ctrl_noaug": "ctrl_noaug",
    "augB_targeted_hard": "augB_targeted_hard",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_matrix",
        default="configs/config_matrix_top3_ctrl_noaug.json",
        help="Any top3 matrix; cfg07 row is extracted by config_id prefix.",
    )
    parser.add_argument("--out_dir", default="configs")
    parser.add_argument("--model_name", default=QWEN35_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    src_path = (project_root / args.source_matrix).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    source_rows = load_json(src_path)
    base_row = None
    for row in source_rows:
        if row.get("parent_config_id") == BASE_CONFIG_ID or row.get("config_id", "").startswith(
            f"{BASE_CONFIG_ID}_"
        ):
            base_row = row
            break
    if base_row is None:
        raise ValueError(f"Could not find cfg07 row derived from {BASE_CONFIG_ID} in {src_path}")

    for strategy, suffix in STRATEGIES.items():
        row = copy.deepcopy(base_row)
        row["model_name"] = args.model_name
        row["model_kwargs"] = {
            "trust_remote_code": True,
            **row.get("model_kwargs", {}),
        }
        row["parent_config_id"] = BASE_CONFIG_ID
        row["augmentation_strategy"] = strategy
        row["base_model_swap"] = {
            "from": "Qwen/Qwen2.5-1.5B-Instruct",
            "to": args.model_name,
        }
        row["config_id"] = (
            "w2_cfg07_lora_qwen35_2b_minimal_lr2e5_a100_" + suffix
        )
        out_path = out_dir / f"config_matrix_final_qwen35_top2_{strategy}.json"
        save_json(out_path, [row])
        print(f"Wrote 1 row -> {out_path} ({row['config_id']})")


if __name__ == "__main__":
    main()
