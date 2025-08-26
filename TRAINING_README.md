# Train a GPT-style model (PyTorch)

This repository contains a minimal training script to train a GPT-style transformer from scratch using PyTorch and Hugging Face.

Files added:
- `train.py` - main training script. Loads tokenizer `ZombitX64/gpt-oss-mini-think` and datasets `ZombitX64/ThaiChatbotConversation` and `HelpingAI/Intermediate-Thinking-130k` (samples 10k entries from the latter).
- `requirements.txt` - Python dependencies.

Quick start (Windows cmd.exe):

1. Create a Python environment and install dependencies:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run training (example):

```cmd
python train.py --epochs 1 --batch-size 4 --output-dir .\model_out
You can select the model implementation with `--model-type simple` (default) or `--model-type hf` to use Hugging Face's GPT2LMHeadModel.
```

Notes and assumptions:
- The script expects public Hugging Face repos for the tokenizer and datasets. It will attempt to download them automatically.
- Token extraction heuristics are conservative; if your datasets use different field names adjust `extract_texts` in `train.py`.
- This is a minimal example for experimentation. For production training, add gradient accumulation, mixed precision (AMP), logging, checkpointing strategies, and evaluation.

Additional frameworks included in requirements:

- TRL (Transformer Reinforcement Learning) for RL fine-tuning and advanced training loops.
- Unsloth — a small utility/framework (may require GitHub install if not on PyPI).

If `unsloth` fails to install from PyPI you can try installing from GitHub (replace the repo with the correct one if different):

```cmd
pip install git+https://github.com/your-org/unsloth.git
```

TRL sometimes requires additional extras (accelerate, bitsandbytes) depending on your setup. Install extras like:

```cmd
pip install "trl[torch]"
```

Cross-lingual tokenizer & training recipe
----------------------------------------

Tokenizer (most important for Thai):

- Approach: SentencePiece Unigram with byte-fallback. Train a Unigram model (not BPE) because Thai benefits from subword segmentation without explicit spaces.
- Vocab size: 80k–100k (recommend ~96k). Include byte fallback to allow any unseen character.
- Normalization: NFC. Normalize varied quotes/dashes to a single canonical form before training.
- User/special tokens: `<bos> <eos> <pad> <nl> <lang_en> <lang_th> <lang_xx> <translate> <copy> <sep>`
- Scripts coverage: Thai, Latin, Devanagari, CJK (subword), emoji, URL, numerals (map Thai↔Arabic as both tokens present).
- Corpus prep: include Thai code-switch examples and noisy spellings (e.g., "เดดไลน์", "deadlineครับ") so tokenizer learns mixed tokens.

Use `tokenizer_train.py` (provided) which wraps `sentencepiece` to train the Unigram tokenizer. Example:

```cmd
python tokenizer_train.py --input data/tokenizer_corpus.txt --model_prefix hanuman_sp --vocab_size 96000
```

Training recipe (high-level):

- Model backbone: 24 layers, d_model=896, heads=14 (GQA=2), RoPE scaled, context=8k.
- FFN: SwiGLU, hidden≈2560 (≈2.85× d_model).
- MoE: 8 experts (FFN experts), top-2 routing, router temperature τ≈1.2, aux loss coef≈0.01.
- Training hyperparams: bf16, AdamW (β1=0.9, β2=0.95), weight_decay=0.1, cosine LR schedule, peak lr=3e-4, warmup=2% steps.
- Sequence lengths: progressive schedule from 2–4k up to 8k.
- Stages: A: 80B tokens (mixture as discussed) → B: 5B Thai LAPT → C: 2–4M instruction pairs (fine-tune)

T4-friendly considerations:

- Micro-batch size 1–2 with large gradient accumulation.
- Use gradient checkpointing and FlashAttention (when available).
- Use ZeRO stage 2/3 (via DeepSpeed or Accelerate integration) to fit larger models on smaller GPUs.
- Use bf16 (if hardware supports it) for faster and memory-efficient training.

Notes on MoE and practical mini-model design:

- Use fewer experts (4–8) and top-1/2 routing to keep inference and training efficient.
- Router should select 1–2 experts per token to preserve capacity while reducing computation.
- Use SwiGLU and RMSNorm for stable training.

Files added:
- `tokenizer_train.py` - helper to train SentencePiece Unigram tokenizer with user symbols.

This README gives the overall plan and helper scripts — adapt paths and cluster-specific settings (DeepSpeed config, storage, data sharding) for your infra.
