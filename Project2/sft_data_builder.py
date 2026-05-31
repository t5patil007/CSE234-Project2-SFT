#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def schema_file_for_db(db_id):
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def load_schema(db_id, schemas_dir):
    path = Path(schemas_dir) / schema_file_for_db(db_id)
    payload = load_json(path)
    table_names = payload["table_names_original"]
    column_names = payload["column_names_original"]
    column_types = payload.get("column_types", [])
    pk_indexes = set(payload.get("primary_keys", []))
    fk_pairs = payload.get("foreign_keys", [])

    table_by_lc = {t.lower(): t for t in table_names}
    cols_by_table = {t: [] for t in table_names}
    col_type_by_table = {t: {} for t in table_names}
    col_by_lc_table = {t: {} for t in table_names}
    col_ref = []

    type_idx = 0
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            col_ref.append(None)
            continue
        table_name = table_names[table_idx]
        col_ref.append((table_name, col_name))
        cols_by_table[table_name].append(col_name)
        if type_idx < len(column_types):
            col_type_by_table[table_name][col_name] = column_types[type_idx]
        type_idx += 1
        col_by_lc_table[table_name][col_name.lower()] = col_name

    pk_cols_by_table = {t: [] for t in table_names}
    for pk_idx in pk_indexes:
        ref = col_ref[pk_idx] if pk_idx < len(col_ref) else None
        if ref is None:
            continue
        table_name, col_name = ref
        pk_cols_by_table[table_name].append(col_name)

    fk_by_table = {t: [] for t in table_names}
    for from_idx, to_idx in fk_pairs:
        from_ref = col_ref[from_idx] if from_idx < len(col_ref) else None
        to_ref = col_ref[to_idx] if to_idx < len(col_ref) else None
        if from_ref is None or to_ref is None:
            continue
        from_table, from_col = from_ref
        to_table, to_col = to_ref
        fk_by_table[from_table].append((from_col, to_table, to_col))

    return {
        "db_id": db_id,
        "tables": table_names,
        "table_by_lc": table_by_lc,
        "cols_by_table": cols_by_table,
        "col_by_lc_table": col_by_lc_table,
        "col_type_by_table": col_type_by_table,
        "pk_cols_by_table": pk_cols_by_table,
        "fk_by_table": fk_by_table,
    }


def tokenize(text):
    out = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return set(out)


def filtered_tables(question, schema, max_tables):
    q_tokens = tokenize(question)
    scored = []
    for table in schema["tables"]:
        table_tokens = tokenize(table)
        overlap = len(q_tokens & table_tokens)
        for col in schema["cols_by_table"].get(table, []):
            col_tokens = tokenize(col)
            overlap += len(q_tokens & col_tokens)
        scored.append((overlap, table))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [table for score, table in scored if score > 0][:max_tables]
    if not chosen:
        chosen = sorted(schema["tables"])[:max_tables]
    return set(chosen)


def serialize_schema(schema, question, schema_mode, filtered_max_tables):
    selected_tables = None
    if schema_mode == "filtered":
        selected_tables = filtered_tables(question, schema, filtered_max_tables)

    lines = []
    for table in sorted(schema["tables"]):
        if selected_tables is not None and table not in selected_tables:
            continue
        cols = schema["cols_by_table"].get(table, [])
        if schema_mode in {"minimal", "filtered"}:
            col_text = ", ".join(cols)
            lines.append(f"{table}({col_text})")
            continue
        if schema_mode == "with_types":
            typed_cols = []
            for col in cols:
                col_type = schema["col_type_by_table"].get(table, {}).get(col, "unknown")
                typed_cols.append(f"{col}:{col_type}")
            lines.append(f"{table}({', '.join(typed_cols)})")
            continue
        if schema_mode == "with_pkfk":
            typed_cols = []
            for col in cols:
                col_type = schema["col_type_by_table"].get(table, {}).get(col, "unknown")
                typed_cols.append(f"{col}:{col_type}")
            pk_cols = sorted(schema["pk_cols_by_table"].get(table, []))
            fk_edges = sorted(schema["fk_by_table"].get(table, []))
            pk_part = f" PK[{', '.join(pk_cols)}]" if pk_cols else " PK[]"
            fk_part = " FK[" + ", ".join(f"{a}->{b}.{c}" for a, b, c in fk_edges) + "]"
            lines.append(f"{table}({', '.join(typed_cols)}){pk_part}{fk_part}")
            continue
        raise ValueError(f"Unsupported schema_mode: {schema_mode}")

    return "\n".join(lines)


def canonicalize_links(links, schema):
    if not isinstance(links, dict):
        return {}
    out = {}
    for raw_table, raw_cols in links.items():
        table = schema["table_by_lc"].get(str(raw_table).lower())
        if not table:
            continue
        col_map = schema["col_by_lc_table"].get(table, {})
        cols = raw_cols if isinstance(raw_cols, list) else []
        seen = set()
        clean = []
        for raw_col in cols:
            col = col_map.get(str(raw_col).lower())
            if col and col not in seen:
                seen.add(col)
                clean.append(col)
        out[table] = sorted(clean)
    return {k: out[k] for k in sorted(out)}


def format_target_json(links, output_style):
    if output_style == "compact":
        return json.dumps(links, separators=(",", ":"))
    if output_style == "pretty":
        return json.dumps(links, indent=2)
    if output_style == "sorted_compact":
        return json.dumps(links, sort_keys=True, separators=(",", ":"))
    if output_style == "sorted_pretty":
        return json.dumps(links, sort_keys=True, indent=2)
    raise ValueError(f"Unsupported output_style: {output_style}")


def format_completion(links, output_style, target_mode):
    payload = format_target_json(links, output_style)
    if target_mode == "json_only":
        return payload
    if target_mode == "reason_then_json":
        tables = list(links.keys())
        col_count = sum(len(cols) for cols in links.values())
        reason = f"Selected tables: {tables}. Selected columns count: {col_count}."
        return f"Reasoning: {reason}\nJSON:\n{payload}"
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def build_prompt(question, db_id, schema_text):
    return [
        {
            "role": "system",
            "content": (
                "You are a schema linker. Return only the requested format for linked "
                "tables and columns."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Database: {db_id}\n"
                f"Question: {question}\n"
                f"Schema:\n{schema_text}\n\n"
                'Output JSON format: {"TableName": ["ColumnName"]}'
            ),
        },
    ]


def build_dataset(
    rows,
    schemas_dir,
    schema_mode,
    output_style,
    target_mode,
    filtered_max_tables,
):
    schema_cache = {}
    out = []
    for row in rows:
        db_id = row["db_id"]
        schema = schema_cache.get(db_id)
        if schema is None:
            schema = load_schema(db_id, schemas_dir)
            schema_cache[db_id] = schema
        canonical_links = canonicalize_links(row.get("schema_links", {}), schema)
        schema_text = serialize_schema(
            schema=schema,
            question=row.get("question", ""),
            schema_mode=schema_mode,
            filtered_max_tables=filtered_max_tables,
        )
        prompt = build_prompt(row.get("question", ""), db_id, schema_text)
        completion_text = format_completion(canonical_links, output_style, target_mode)
        out.append(
            {
                "question_id": row.get("question_id"),
                "db_id": db_id,
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": completion_text}],
                "meta": {
                    "schema_mode": schema_mode,
                    "output_style": output_style,
                    "target_mode": target_mode,
                },
            }
        )
    return out


def build_output_path(base_dir, split_name, schema_mode, output_style, target_mode):
    name = f"sft_{split_name}_{schema_mode}_{output_style}_{target_mode}.json"
    return str(Path(base_dir) / name)


def run_single(args):
    rows = load_json(args.input_json)
    payload = build_dataset(
        rows=rows,
        schemas_dir=args.schemas_dir,
        schema_mode=args.schema_mode,
        output_style=args.output_style,
        target_mode=args.target_mode,
        filtered_max_tables=args.filtered_max_tables,
    )
    save_json(args.output_json, payload)
    print(f"Wrote {len(payload)} rows to {args.output_json}")


def run_grid(args):
    train_rows = load_json(args.train_json)
    val_rows = load_json(args.validation_json)
    out_dir = Path(args.variants_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_modes = ["minimal", "with_types", "with_pkfk", "filtered"]
    output_styles = ["sorted_compact", "sorted_pretty"]
    target_modes = ["json_only", "reason_then_json"]
    for split_name, rows in [("train", train_rows), ("validation", val_rows)]:
        for schema_mode in schema_modes:
            for output_style in output_styles:
                for target_mode in target_modes:
                    payload = build_dataset(
                        rows=rows,
                        schemas_dir=args.schemas_dir,
                        schema_mode=schema_mode,
                        output_style=output_style,
                        target_mode=target_mode,
                        filtered_max_tables=args.filtered_max_tables,
                    )
                    out_path = build_output_path(
                        base_dir=out_dir,
                        split_name=split_name,
                        schema_mode=schema_mode,
                        output_style=output_style,
                        target_mode=target_mode,
                    )
                    save_json(out_path, payload)
                    print(f"Wrote {len(payload)} rows to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--validation_json", default="validation.json")
    parser.add_argument("--variants_out_dir", default="sft_variants")
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument(
        "--schema_mode",
        choices=["minimal", "with_types", "with_pkfk", "filtered"],
        default="minimal",
    )
    parser.add_argument(
        "--output_style",
        choices=["compact", "pretty", "sorted_compact", "sorted_pretty"],
        default="sorted_compact",
    )
    parser.add_argument(
        "--target_mode",
        choices=["json_only", "reason_then_json"],
        default="json_only",
    )
    parser.add_argument("--filtered_max_tables", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "single":
        if not args.input_json or not args.output_json:
            raise ValueError("single mode requires --input_json and --output_json")
    return args


def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
