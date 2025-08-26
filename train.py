import os
import argparse
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from modeling import build_from_config
import torch
from unsloth_helper import apply_unsloth_augmentations
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def extract_texts(split_ds):
    # Try common fields to extract human-readable text
    candidates = ['text', 'content', 'conversation', 'dialogue', 'messages', 'prompt', 'instruction', 'response']
    texts = []
    for item in split_ds:
        for c in candidates:
            if c in item and item[c]:
                # if conversation is list or dict, try to flatten
                val = item[c]
                if isinstance(val, list):
                    texts.append(' '.join(map(str, val)))
                else:
                    texts.append(str(val))
                break
        else:
            # fallback: try join all string fields
            joined = ' '.join(str(v) for v in item.values() if isinstance(v, str) and v.strip())
            if joined:
                texts.append(joined)
    return texts


def prepare_datasets(tokenizer_name, sample_intermediate=10000, seq_len=256):
    # backward compatible wrapper kept for single-flag calls
    return prepare_datasets(tokenizer_name, sample_intermediate=sample_intermediate, seq_len=seq_len, use_unsloth=False)


def prepare_datasets(tokenizer_name, sample_intermediate=10000, seq_len=256, use_unsloth=False):
    print('Loading tokenizer:', tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load datasets from HF repos
    print('Loading datasets...')
    ds1 = load_dataset('ZombitX64/ThaiChatbotConversation')
    ds2 = load_dataset('HelpingAI/Intermediate-Thinking-130k')

    texts = []
    # extract from ds1 (take train if available else first split)
    split1 = next(iter(ds1.values()))
    texts += extract_texts(split1)

    # sample 10k from ds2 if large
    split2 = next(iter(ds2.values()))
    if len(split2) > sample_intermediate:
        split2 = split2.shuffle(seed=42).select(range(sample_intermediate))
    texts += extract_texts(split2)

    # optional data augmentation using Unsloth if available
    if use_unsloth:
        try:
            texts = apply_unsloth_augmentations(texts)
            print(f'Applied Unsloth augmentations, total texts now {len(texts)}')
        except Exception as e:
            logging.warning('Failed to apply Unsloth augmentations: %s', e)

    print(f'Collected {len(texts)} text pieces')

    # create a HF Dataset for tokenization
    hf = Dataset.from_dict({'text': texts})

    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=seq_len)

    hf = hf.map(tokenize_fn, batched=False, remove_columns=['text'])

    # convert to torch tensors via collate_fn at DataLoader time
    return tokenizer, hf


def collate_fn(batch, tokenizer):
    input_ids = [b['input_ids'] for b in batch]
    # tokenizer.pad will handle padding and return tensors
    return tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')


def train(args):
    tokenizer, hf = prepare_datasets(args.tokenizer, sample_intermediate=args.sample_intermediate, seq_len=args.seq_len, use_unsloth=args.use_unsloth)

    # ensure special effort tokens exist in the tokenizer before model instantiation
    special_think_tokens = ['<think_none>', '<think_short>', '<think_medium>', '<think_long>', '<scratch>', '</scratch>', '<final>', '<verify>', '<tool>']
    added = [t for t in special_think_tokens if t not in tokenizer.get_vocab()]
    if added:
        tokenizer.add_special_tokens({'additional_special_tokens': added})

    # Prepare DataLoader
    dataloader = DataLoader(hf, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    # Model config and instantiation (from-scratch GPT2-style)
    vocab_size = len(tokenizer)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id or 0,
        eos_token_id=tokenizer.eos_token_id or 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    # attach optional Hanuman-specific flags to config so build_from_config can consume them
    setattr(config, 'use_think_head', args.use_think_head)
    setattr(config, 'think_aux_coef', args.think_aux_coef)

    if args.model_type == 'hf':
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(vocab_size)
    else:
        # build our lightweight SimpleGPT implementation
        model = build_from_config(config)
        # tie weights if possible (embedding -> head)
        try:
            model.head.weight = model.wte.weight
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # map think tokens to ids on the model for runtime tag embedding usage
    try:
        think_map = {
            'none': tokenizer.convert_tokens_to_ids('<think_none>'),
            'short': tokenizer.convert_tokens_to_ids('<think_short>'),
            'medium': tokenizer.convert_tokens_to_ids('<think_medium>'),
            'long': tokenizer.convert_tokens_to_ids('<think_long>'),
        }
        model.think_token_ids = think_map
    except Exception:
        model.think_token_ids = {}

    # If user requested TRL integration, provide a usage hint/example file and require HF model
    if args.use_trl:
        if args.model_type != 'hf':
            raise RuntimeError('--use-trl requires using the HuggingFace model implementation: --model-type hf')
        try:
            import trl  # noqa: F401
        except Exception:
            print('\nTRL package not importable in this environment.\n')
            print('A TRL usage example has been written to trl_example.py in the repo root.\n')
            print('Install TRL with `pip install trl[torch]` and follow that example to run PPO-style fine-tuning.')
            # continue to allow supervised training path
        else:
            print('TRL detected. You can run reinforcement learning loops using TRL; see trl_example.py for a minimal example.')

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f'epoch {epoch+1}/{args.epochs}')
        running_loss = 0.0
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            # primary LM loss (may be None for some custom forwards)
            loss = getattr(outputs, 'loss', None)

            # collect MoE auxiliary losses from model blocks if present
            moe_aux = None
            try:
                moe_losses = []
                if hasattr(model, 'blocks'):
                    for blk in model.blocks:
                        mlp = getattr(blk, 'mlp', None)
                        if mlp is not None and hasattr(mlp, 'last_aux_loss') and mlp.last_aux_loss is not None:
                            val = mlp.last_aux_loss
                            if not isinstance(val, torch.Tensor):
                                val = torch.tensor(float(val), device=device)
                            else:
                                val = val.to(device)
                            moe_losses.append(val)
                if moe_losses:
                    moe_aux = sum(moe_losses)
            except Exception:
                moe_aux = None

            # include thought auxiliary loss if present and primary loss didn't already include it
            thought_added = False
            if hasattr(outputs, 'thought_loss') and outputs.thought_loss is not None:
                # model.forward may already have folded scaled thought loss into outputs.loss.
                # Only add it here if outputs.loss is None.
                if loss is None:
                    tl = outputs.thought_loss
                    if not isinstance(tl, torch.Tensor):
                        tl = torch.tensor(float(tl), device=device)
                    else:
                        tl = tl.to(device)
                    tl = tl * getattr(model, 'think_aux_coef', 1.0)
                    thought_added = True
                else:
                    tl = None

            # build final scalar/tensor loss for backward
            if loss is None:
                if thought_added:
                    final_loss = tl
                    if moe_aux is not None:
                        final_loss = final_loss + moe_aux
                elif moe_aux is not None:
                    final_loss = moe_aux
                else:
                    # as a fallback, create a zero-tensor to avoid crash
                    final_loss = torch.tensor(0.0, device=device)
            else:
                # ensure loss tensor is on the right device
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(float(loss), device=device)
                else:
                    loss = loss.to(device)
                final_loss = loss
                if moe_aux is not None:
                    final_loss = final_loss + moe_aux

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss else loss.item()
            loop.set_postfix(loss=running_loss)

        # save checkpoint each epoch
        out = os.path.join(args.output_dir, f'epoch-{epoch+1}')
        os.makedirs(out, exist_ok=True)
        # save with safetensors option if requested
        try:
            model.save_pretrained(out, use_safetensors=args.use_safetensors)
        except TypeError:
            # HF models accept save_pretrained but not our flag; handle HF separately
            if args.use_safetensors:
                try:
                    from safetensors.torch import save_file as safe_save
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    safe_save(state, os.path.join(out, 'pytorch_model.safetensors'))
                except Exception:
                    model.save_pretrained(out)
            else:
                model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print('Saved checkpoint to', out)


def main():
    parser = argparse.ArgumentParser(description='Train a GPT-style model from scratch using provided tokenizer and datasets')
    parser.add_argument('--tokenizer', type=str, default='ZombitX64/gpt-oss-mini-think')
    parser.add_argument('--sample-intermediate', type=int, default=10000, dest='sample_intermediate')
    parser.add_argument('--seq-len', type=int, default=256, dest='seq_len')
    parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n-embd', type=int, default=512, dest='n_embd')
    parser.add_argument('--n-layer', type=int, default=6, dest='n_layer')
    parser.add_argument('--n-head', type=int, default=8, dest='n_head')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--model-type', type=str, default='simple', choices=['simple', 'hf'], help='Which model implementation to use')
    parser.add_argument('--use-unsloth', action='store_true', help='Enable Unsloth-based data augmentations when available')
    parser.add_argument('--use-trl', action='store_true', help='Enable TRL integration hints (requires --model-type hf for RL flows)')
    parser.add_argument('--use-safetensors', action='store_true', help='Save model weights as .safetensors when possible')
    parser.add_argument('--use-think-head', action='store_true', help='Enable think head auxiliary outputs and loss')
    parser.add_argument('--think-aux-coef', type=float, default=1.0, help='Scaling coefficient for think head auxiliary loss')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
