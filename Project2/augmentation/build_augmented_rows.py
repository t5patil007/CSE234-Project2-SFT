#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data_builder import canonicalize_links, load_schema as load_sft_schema

try:
    from sql_to_schema_links import extract_schema_links, load_schema as load_sql_schema
except Exception:
    extract_schema_links = None
    load_sql_schema = None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["augA_local_sql", "augB_targeted_hard", "augC_hybrid_small"],
    )
    parser.add_argument("--max_new_rows", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_aug_in", default=None)
    parser.add_argument("--api_max_rows", type=int, default=25)
    parser.add_argument("--output_dir", default="augmentation/data")
    parser.add_argument("--provenance_out", default="augmentation/augmentation_provenance.jsonl")
    parser.add_argument("--dialect", default="tsql")
    return parser.parse_args()


def question_rewrite(question, mode):
    q = question.strip()
    templates = {
        "a": [
            "Identify the relevant tables and columns for this request: {q}",
            "For this database question, return the linked schema elements: {q}",
            "Find schema links (tables/columns) for: {q}",
        ],
        "b": [
            "This question involves multiple relations; identify linked schema items: {q}",
            "Determine tables and columns needed to answer: {q}",
            "Schema-link this harder query carefully: {q}",
        ],
        "c": [
            "Provide schema links for the following user request: {q}",
            "Map this request to schema entities (table -> columns): {q}",
            "Identify the SQL-referenced schema pieces for: {q}",
        ],
    }
    choices = templates[mode]
    return random.choice(choices).format(q=q)


def is_near_duplicate(norm_question, seen_questions, threshold=0.94):
    for prev in seen_questions:
        if SequenceMatcher(None, norm_question, prev).ratio() >= threshold:
            return True
    return False


def build_candidates(train_rows, strategy):
    db_counts = Counter(r["db_id"] for r in train_rows)
    underrepresented = {db for db, c in db_counts.items() if c <= 6}

    candidates = []
    if strategy == "augA_local_sql":
        for r in train_rows:
            links = r.get("schema_links", {})
            if not isinstance(links, dict) or not links:
                continue
            if len(links) <= 2:
                candidates.append((r, "a", "local"))
    elif strategy == "augB_targeted_hard":
        for r in train_rows:
            links = r.get("schema_links", {})
            if not isinstance(links, dict) or not links:
                continue
            multi_table = len(links) >= 2
            has_empty_cols = any(isinstance(cols, list) and len(cols) == 0 for cols in links.values())
            if r["db_id"] in underrepresented or multi_table or has_empty_cols:
                candidates.append((r, "b", "local"))
    elif strategy == "augC_hybrid_small":
        for r in train_rows:
            links = r.get("schema_links", {})
            if isinstance(links, dict) and links:
                candidates.append((r, "c", "local"))
    return candidates


def build_api_candidates(api_rows):
    out = []
    for row in api_rows:
        if not isinstance(row, dict):
            continue
        db_id = row.get("db_id")
        question = row.get("question")
        if not db_id or not question:
            continue
        out.append(
            {
                "question_id": row.get("question_id"),
                "db_id": db_id,
                "question": question,
                "gold_sql": row.get("gold_sql"),
                "schema_links": row.get("schema_links"),
                "source": "vendor_api",
                "vendor": row.get("vendor", "unspecified"),
            }
        )
    return out


def main():
    args = parse_args()
    random.seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    train_path = (project_root / args.train_json).resolve()
    schemas_dir = (project_root / args.schemas_dir).resolve()
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_json(train_path)
    if not isinstance(train_rows, list):
        raise ValueError(f"Expected list in train_json, got {type(train_rows).__name__}")

    sql_schema_cache = {}
    sft_schema_cache = {}

    def get_sql_schema(db_id):
        if load_sql_schema is None:
            raise RuntimeError("sql_to_schema_links dependencies unavailable.")
        if db_id not in sql_schema_cache:
            sql_schema_cache[db_id] = load_sql_schema(str(schemas_dir), db_id)
        return sql_schema_cache[db_id]

    def get_sft_schema(db_id):
        if db_id not in sft_schema_cache:
            sft_schema_cache[db_id] = load_sft_schema(db_id, str(schemas_dir))
        return sft_schema_cache[db_id]

    candidate_pool = build_candidates(train_rows, args.strategy)
    random.shuffle(candidate_pool)

    api_candidates = []
    if args.strategy == "augC_hybrid_small" and args.api_aug_in:
        api_in = Path(args.api_aug_in).resolve()
        if api_in.exists():
            api_candidates = build_api_candidates(load_json(api_in))
            random.shuffle(api_candidates)

    max_new_rows = max(1, args.max_new_rows)
    api_quota = min(args.api_max_rows, max_new_rows // 3) if args.strategy == "augC_hybrid_small" else 0
    local_quota = max_new_rows - api_quota

    max_qid = max((int(r.get("question_id", 0)) for r in train_rows), default=0)
    next_qid = max_qid + 1

    seen_signatures = set()
    seen_q_by_db = defaultdict(list)
    for r in train_rows:
        db = r.get("db_id")
        q = normalize_text(r.get("question", ""))
        links = r.get("schema_links", {})
        sig = (db, q, json.dumps(links, sort_keys=True))
        seen_signatures.add(sig)
        seen_q_by_db[db].append(q)

    augmented_rows = []
    provenance_rows = []
    stats = Counter()

    def validate_and_add(row, source, generator_tag, vendor=None):
        nonlocal next_qid
        db_id = row["db_id"]
        question = row["question"]
        gold_sql = row.get("gold_sql")
        existing_links = row.get("schema_links")

        links = None
        if isinstance(gold_sql, str) and gold_sql.strip() and extract_schema_links is not None:
            links, err = extract_schema_links(gold_sql, get_sql_schema(db_id), dialect=args.dialect)
            if err:
                stats["skip_parse_error"] += 1
                return
        elif isinstance(gold_sql, str) and gold_sql.strip() and extract_schema_links is None:
            if isinstance(existing_links, dict):
                links = existing_links
                stats["fallback_existing_links_no_sql_parser"] += 1
            else:
                stats["skip_no_sql_parser_and_no_links"] += 1
                return
        elif isinstance(existing_links, dict):
            links = existing_links
        else:
            stats["skip_no_links"] += 1
            return

        canonical_links = canonicalize_links(links, get_sft_schema(db_id))
        norm_q = normalize_text(question)
        sig = (db_id, norm_q, json.dumps(canonical_links, sort_keys=True))

        if sig in seen_signatures:
            stats["skip_exact_duplicate"] += 1
            return
        if is_near_duplicate(norm_q, seen_q_by_db[db_id]):
            stats["skip_near_duplicate"] += 1
            return

        out_row = {
            "question_id": next_qid,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql if isinstance(gold_sql, str) else "",
            "schema_links": canonical_links,
            "augmentation_meta": {
                "strategy": args.strategy,
                "source": source,
                "generator_tag": generator_tag,
                "vendor": vendor,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        next_qid += 1

        augmented_rows.append(out_row)
        seen_signatures.add(sig)
        seen_q_by_db[db_id].append(norm_q)
        stats["accepted"] += 1

        provenance_rows.append(
            {
                "question_id": out_row["question_id"],
                "strategy": args.strategy,
                "source": source,
                "vendor": vendor,
                "generator_tag": generator_tag,
                "db_id": db_id,
                "question": question,
                "gold_sql_present": bool(out_row["gold_sql"]),
                "used_sql_to_schema_links": bool(out_row["gold_sql"]),
                "external_api_role": (
                    "annotation_or_generation_only_not_training_model"
                    if source == "vendor_api"
                    else "none"
                ),
                "timestamp_utc": out_row["augmentation_meta"]["timestamp_utc"],
            }
        )

    for row, mode, _source in candidate_pool:
        if stats["accepted"] >= local_quota:
            break
        candidate = dict(row)
        candidate["question"] = question_rewrite(row["question"], mode)
        validate_and_add(candidate, source="local", generator_tag=f"rewrite_{mode}")

    for api_row in api_candidates:
        if stats["accepted"] >= max_new_rows:
            break
        if len([r for r in augmented_rows if r["augmentation_meta"]["source"] == "vendor_api"]) >= api_quota:
            break
        validate_and_add(
            api_row,
            source="vendor_api",
            generator_tag="api_hybrid_import",
            vendor=api_row.get("vendor", "unspecified"),
        )

    aug_rows_path = out_dir / f"aug_rows_{args.strategy}.json"
    merged_train_path = out_dir / f"augmented_train_{args.strategy}.json"
    report_path = out_dir / f"augmentation_report_{args.strategy}.json"
    strategy_prov_path = out_dir / f"provenance_{args.strategy}.jsonl"

    save_json(aug_rows_path, augmented_rows)
    save_json(merged_train_path, train_rows + augmented_rows)

    report = {
        "strategy": args.strategy,
        "seed": args.seed,
        "max_new_rows": max_new_rows,
        "requested_api_max_rows": args.api_max_rows,
        "accepted_rows": stats["accepted"],
        "accepted_local_rows": len([r for r in augmented_rows if r["augmentation_meta"]["source"] == "local"]),
        "accepted_vendor_api_rows": len([r for r in augmented_rows if r["augmentation_meta"]["source"] == "vendor_api"]),
        "skip_exact_duplicate": stats["skip_exact_duplicate"],
        "skip_near_duplicate": stats["skip_near_duplicate"],
        "skip_parse_error": stats["skip_parse_error"],
        "skip_no_links": stats["skip_no_links"],
        "aug_rows_path": str(aug_rows_path),
        "augmented_train_path": str(merged_train_path),
        "strategy_provenance_path": str(strategy_prov_path),
        "aggregate_provenance_path": str((project_root / args.provenance_out).resolve()),
        "external_api_boundary": (
            "external_vendor_api_may_be_used_for_annotation_or_generation_only_never_as_final_trained_model"
        ),
        "sql_parser_available": extract_schema_links is not None,
    }
    save_json(report_path, report)

    append_jsonl(strategy_prov_path, provenance_rows)
    append_jsonl((project_root / args.provenance_out).resolve(), provenance_rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
