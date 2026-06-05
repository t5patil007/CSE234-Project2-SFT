# CSE234 Project 2 — Schema Linking (SFT)

## Running inference (grader / submission)

From this directory

```bash
python3 main.py --input <questions.json> --output <predictions.json>
```

Optional flags:

- `--schemas_dir ./schemas` — schema JSON directory (default; resolved from the **current working directory**)
- `--checkpoint_path <path>` — LoRA adapter directory containing `adapter_config.json` (default: bundled adapter under `adapter/...`)
### Local validation (optional)

```bash
python3 main.py --input validation_input.json --output preds.json
python eval.py --predictions preds.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas/ \
  --questions_input validation_input.json
```

## Environment

**Inference requires:**

- `torch`
- `transformers`
- `peft`
- `accelerate` (pulled in by transformers device_map)


**Data augmentation helper** (`sql_to_schema_links.py`): `sqlglot>=23.0`

The base model weights are downloaded from Hugging Face on first run (`Qwen/Qwen2.5-1.5B-Instruct` for the default adapter).
