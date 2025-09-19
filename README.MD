# FinAI2025 — LoRA Adapter for Task II (FinGPT Agents in Real Life)

**Team / Author:** SatishChandraPhD and Dr. G. Balakrishna 
**Contest:** FinAI_Contest_2025 — Task II (FinGPT Agents in Real Life)

**Short summary:**  
# FinAI2025 — LoRA Fine-tuned Llama 3.1 8B Instruct

This repository contains the submission for **FinAI 2025 Contest — Task 2 (FinGPT Agent)**.  
We fine-tuned **Meta Llama 3.1 8B Instruct** with **LoRA adapters** on financial tasks spanning:

- **Bloomberg-style financial analytics**
- **CFA-style quantitative reasoning**
- **XBRL structured reporting questions**

Only the **LoRA adapter weights** are included here (not the full base model).

---

## Model Card

- **Base model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Fine-tuning method**: Parameter-Efficient Fine-Tuning (LoRA, rank=16)
- **Dataset**: Mixture of curated financial tasks + synthetic augmentation (CFA, Bloomberg, XBRL)
- **Hardware**: NVIDIA A100 (40GB), single-GPU
- **Intended use**: Financial reasoning, analytics, reporting tasks (contest evaluation)
- **Limitations**: Not a production trading system. Outputs may be incomplete or inconsistent.

---

## Quick Usage

### 1) Install dependencies

```bash
pip install -r requirements.txt

2) Run single prompt inference
python inference.py \
  --adapter ./targeted_lora_adapter \
  --prompt "Compute the Debt-to-Equity ratio given liabilities=200, equity=400."
3) Run on a JSONL test file
{"context": "Compute D/E given liabilities=522, equity=785.", "task": "Bloomberg"}
{"context": "In XBRL, 'NetIncomeLoss' shows negative values. What should analyst check?", "task": "XBRL"}

# File Structure
.
├── inference.py               # Minimal inference script
├── requirements.txt           # Dependencies
├── targeted_lora_adapter/     # LoRA adapter weights
├── README.md                  # This file

# Citation
If you use this work, please cite the FinAI 2025 Contest.
# License
Apache 2.0 (matching base model + datasets license).


