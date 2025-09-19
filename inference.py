#!/usr/bin/env python3
"""
Minimal inference script for LoRA adapter + quantized base model.
Saves per-example predictions to a JSONL file (one JSON object per line).

Usage examples:
  # single prompt
  python inference.py --base_model meta-llama/Llama-3.1-8B-Instruct \
                      --adapter_dir ./finai_lora_adapter \
                      --prompt "Explain the impact of inflation on bond yields." \
                      --output ./single_result.jsonl

  # JSONL input (each line: {"context": "...", "target": "...", "task": "..."})
  python inference.py --base_model meta-llama/Llama-3.1-8B-Instruct \
                      --adapter_dir ./finai_lora_adapter \
                      --input_file ./test_dataset.norm.jsonl \
                      --output ./test_results.jsonl \
                      --batch_size 4 \
                      --max_new_tokens 128 \
                      --quantize 4bit

Notes:
 - This script expects the LoRA adapter folder (adapter_dir) that was saved via PEFT (adapter + tokenizer files).
 - If you upload to Hugging Face, only push the adapter dir + tokenizer files + this script (do NOT upload base model weights).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
# BitsAndBytesConfig is in transformers_bitsandbytes integration
from transformers import BitsAndBytesConfig

# ---------- helpers ----------
def read_input_lines(path: Path):
    """Load JSONL or plain text. If JSONL, each line should be a dict with at least 'context'."""
    data = []
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} not found")
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # try JSON
            try:
                obj = json.loads(ln)
                # normalize keys
                if isinstance(obj, dict):
                    context = obj.get("context") or obj.get("prompt") or obj.get("question") or ""
                    target = obj.get("target") or obj.get("answer") or ""
                    task = obj.get("task") or obj.get("meta") or ""
                    data.append({"context": context, "target": target, "task": task})
                    continue
            except Exception:
                pass
            # fallback: treat whole line as context
            data.append({"context": ln, "target": "", "task": ""})
    return data

def save_jsonl(items: List[Dict], outpath: Path):
    with outpath.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Run inference with base model + LoRA adapter (quantized friendly).")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model id (HF) or local path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to saved LoRA adapter directory (PEFT).")
    parser.add_argument("--input_file", type=str, default=None,
                        help="JSONL input (one JSON object per line with 'context' key).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt string (use --prompt or --input_file).")
    parser.add_argument("--output", type=str, default="./test_results.jsonl",
                        help="Output JSONL file with per-example predictions.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu'. If omitted, auto-select.")
    parser.add_argument("--quantize", type=str, default="4bit", choices=["4bit", "8bit", "none"],
                        help="Use quantized base model load (4bit recommended).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling if provided.")
    parser.add_argument("--left_pad", action="store_true", help="Force tokenizer.padding_side='left' for decoder-only models.")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    # If adapter saved tokenizer not provided, fallback to base_model's tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Ensure pad token for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.left_pad:
        tokenizer.padding_side = "left"

    # Prepare quantization config if requested
    quant_config = None
    if args.quantize == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )
        print("Using 4-bit quantization config.")
    elif args.quantize == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization config.")
    else:
        quant_config = None
        print("No quantization requested (full precision).")

    # Load base model (quantized if quant_config provided)
    # Use device_map='auto' for multi-GPU/A100 offloading; this allows accelerate to place weights optimally.
    from transformers import AutoConfig
    print("Loading base model:", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if quant_config and quant_config.bnb_4bit_compute_dtype == torch.bfloat16 else None,
    )

    # Attach PEFT adapter
    print("Attaching LoRA adapter from:", args.adapter_dir)
    model = PeftModel.from_pretrained(model, args.adapter_dir, device_map="auto" if device.type == "cuda" else None)
    model.eval()

    # Load inputs
    inputs = []
    if args.prompt:
        inputs.append({"context": args.prompt, "target": "", "task": ""})
    elif args.input_file:
        inputs = read_input_lines(Path(args.input_file))
    else:
        raise ValueError("Either --prompt or --input_file must be provided.")

    print(f"Loaded {len(inputs)} examples.")

    # Batched generation
    results = []
    batch_size = max(1, args.batch_size)
    for i in tqdm(range(0, len(inputs), batch_size), desc="Batched inference"):
        batch = inputs[i: i + batch_size]
        contexts = [ex["context"] for ex in batch]
        # Tokenize
        enc = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        # move to device (note: model is device-mapped when using accelerate; in that case the tensors should remain on CPU)
        for k, v in enc.items():
            enc[k] = v.to(device)

        # Generation params
        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False
        )

        # Remove invalid keys for specific transformers versions (some warnings may appear)
        try:
            outputs = model.generate(**gen_kwargs)
        except Exception as e:
            # Fallback: move model to device explicitly and try again (only for CPU or single-GPU flows)
            print("Generation raised:", e)
            # attempt to place tensors on model device then generate
            try:
                model_device = next(model.parameters()).device
                for k in enc:
                    enc[k] = enc[k].to(model_device)
                gen_kwargs["input_ids"] = enc["input_ids"]
                gen_kwargs["attention_mask"] = enc.get("attention_mask", None)
                outputs = model.generate(**gen_kwargs)
            except Exception as e2:
                raise RuntimeError(f"Generation failed twice: {e} ; {e2}")

        # Decode per-example
        for idx, out in enumerate(outputs):
            # For some model/generation variants out may be tensors or lists
            if torch.is_tensor(out):
                decoded = tokenizer.decode(out.tolist(), skip_special_tokens=True)
            else:
                try:
                    decoded = tokenizer.decode(out, skip_special_tokens=True)
                except Exception:
                    decoded = str(out)
            item = batch[idx].copy()
            item["prediction"] = decoded
            results.append(item)

    # Save results
    outpath = Path(args.output)
    save_jsonl(results, outpath)
    print("Saved detailed per-example results to", outpath)

if __name__ == "__main__":
    main()
