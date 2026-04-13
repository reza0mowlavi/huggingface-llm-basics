# HuggingFace + LLM Workshop (GSM8K)

This repo contains a sequence of Jupyter notebooks that introduce core HuggingFace concepts (models, tokenizers, Trainer), inference patterns, and parameter‑efficient fine‑tuning (LoRA/QLoRA) using the GSM8K dataset.

## Structure
All course material lives in `notebooks/` and is intended to be followed in order:

1. `notebooks/1.Basics of HuggingFace.ipynb`
   - How `from_pretrained` works (config vs weights)
   - AutoModel vs task heads (`AutoModelForCausalLM`, `AutoModelForSequenceClassification`, etc.)
   - Implementing a minimal custom `PreTrainedModel` and saving/loading it
   - A quick Trainer intro on a toy classifier

2. `notebooks/2.Tokenizer and BasicGeneration.ipynb`
   - Tokenizers, special tokens, chat templates
   - Greedy decoding vs sampling
   - Streaming/step‑by‑step generation

3. `notebooks/3.Batching with HuggingFace.ipynb`
   - GSM8K inference with batched generation
   - Left padding for decoder‑only models

4. `notebooks/4.Batching with vLLM.ipynb`
   - vLLM runtime basics
   - Batch serving and performance‑oriented settings

5. `notebooks/5.Supervised Finetuning LLama1B.ipynb`
   - End‑to‑end SFT using `Trainer`
   - Prompt masking and custom collator
   - Save/reload and evaluate

6. `notebooks/6.Supervised Finetuning with LoRA.ipynb`
   - Parameter‑efficient fine‑tuning with PEFT LoRA
   - Saving and loading adapters
   - vLLM inference with LoRA adapters

7. `notebooks/7.Supervised Finetuning with QLoRA.ipynb`
   - 4‑bit quantization with bitsandbytes
   - QLoRA training pipeline
   - Adapter save/reload and evaluation

## Dataset
GSM8K is used throughout. For consistency and offline‑friendly runs, the notebooks include local JSONL files in `notebooks/gsm8k/`.

## Utilities
Shared helper functions live in `notebooks/utils.py`:
- GSM8K formatting and chat templates
- Prompt masking for SFT
- Batched generation utilities
- Accuracy computation

## Environment Notes
- `notebooks/1.Basics of HuggingFace.ipynb`, `notebooks/2.Tokenizer and BasicGeneration.ipynb`, and `notebooks/3.Batching with HuggingFace.ipynb` can run on Colab. The later notebooks become too time‑consuming to finish within a few minutes on Colab, so participants can mainly follow Parts 1–2, and optionally Part 3 (maybe) and Part 4 (maybe).
- LoRA/QLoRA notebooks assume CUDA availability; QLoRA requires `bitsandbytes`.
- vLLM requires a compatible CUDA GPU.

If you want a minimal environment, install:
- `torch`
- `transformers`
- `datasets`
- `peft`
- `vllm` (for Part 4)
- `bitsandbytes` (for Part 7)

## Outputs
Model checkpoints and adapters are saved under `notebooks/artifacts/`.
