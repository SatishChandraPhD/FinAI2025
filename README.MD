# FinAI2025 — LoRA Adapter for Task II (FinGPT Agents in Real Life)

**Team / Author:** SatishChandraPhD  
**Contest:** FinAI_Contest_2025 — Task II (FinGPT Agents in Real Life)

**Short summary:**  
This repository contains a small LoRA adapter and a minimal inference script to run the adapter on a quantized base model (meta-llama/Llama-3.1-8B-Instruct). The repository is intentionally light-weight — **we do not upload base model weights** (only the LoRA adapter + tokenizer + inference code). This matches the contest submission rules.

---

## Repo contents (what to upload)

- `adapter/` — **REQUIRED** folder containing the saved PEFT/LoRA adapter (e.g. `adapter_config.json`, adapter weights, optionally tokenizer files).
  - **Do not** upload base model full weights here.
- `inference.py` — Minimal single-file inference script (usage below).
- `requirements.txt` — Python dependencies.
- `example_test.jsonl` — Small JSONL file (2–10 examples) demonstrating input format.
- `README.md` (this file).
- Optional: `model_card.md` or `model_card.json` for Hugging Face model page.

---

## Quick usage (example)

### 1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
