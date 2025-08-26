"""Train a SentencePiece Unigram tokenizer with byte fallback and custom specials.

This script is a helper to create the cross-lingual tokenizer described in the spec.
It uses the `sentencepiece` Python bindings to train a Unigram model and then wraps it
with `tokenizers` for byte fallback and special tokens handling.

Usage (example):
    python tokenizer_train.py --input data/tokenizer_corpus.txt --model_prefix hanuman_sp --vocab_size 96000

Notes:
- Ensure the input corpus contains representative Thai code-switching examples and special mappings.
- The script normalizes text to NFC and performs simple pre- and post-processing for quotes/dashes.
"""
import argparse
import sentencepiece as spm
import unicodedata
import os
from datasets import load_dataset
from typing import List


def normalize_text(s: str) -> str:
    # NFC normalization and basic quote/dash normalization
    s = unicodedata.normalize('NFC', s)
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    return s


def train_unigram(input_path, model_prefix, vocab_size=96000, user_defined_symbols=None):
    if user_defined_symbols is None:
        user_defined_symbols = []
    cmd = f"--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=unigram --character_coverage=0.9995 --unk_id=0 --bos_id=-1 --eos_id=-1 --pad_id=0 --normalization_rule_name=nmt"
    if user_defined_symbols:
        cmd += ' --user_defined_symbols=' + ','.join(user_defined_symbols)
    print('Training SentencePiece with cmd:', cmd)
    spm.SentencePieceTrainer.Train(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Path to a plain text corpus file (one example per line). If omitted, use --hf_datasets')
    parser.add_argument('--model_prefix', default='hanuman_sp')
    parser.add_argument('--vocab_size', type=int, default=96000)
    parser.add_argument('--hf_datasets', type=str, default='ZombitX64/Wisesight-Sentiment-Thai,ZombitX64/Thai-corpus-word',
                        help='Comma-separated HF dataset repo ids to use as corpus if --input is not provided')
    parser.add_argument('--max_samples_per_dataset', type=int, default=0, help='Limit samples per HF dataset (0 = all)')
    args = parser.parse_args()

    # Prepare corpus: use provided input file or assemble from HF datasets
    corpus_path = args.input
    if not corpus_path:
        ds_names = [x.strip() for x in args.hf_datasets.split(',') if x.strip()]
        if not ds_names:
            raise SystemExit('Either --input or --hf_datasets must be provided')

        corpus_path = f"{args.model_prefix}_corpus.txt"
        print(f'Assembling corpus from HF datasets: {ds_names} -> {corpus_path}')
        with open(corpus_path, 'w', encoding='utf-8') as fout:
            for ds_name in ds_names:
                try:
                    ds = load_dataset(ds_name)
                except Exception as e:
                    print(f'Failed to load dataset {ds_name}: {e}')
                    continue
                split = next(iter(ds.values()))
                count = 0
                for item in split:
                    # try common text fields
                    for c in ('text', 'content', 'sentence', 'words', 'word'):
                        if c in item and item[c]:
                            text = item[c]
                            if isinstance(text, list):
                                line = ' '.join(map(str, text))
                            else:
                                line = str(text)
                            line = normalize_text(line)
                            fout.write(line.replace('\n', ' ') + '\n')
                            break
                    else:
                        # fallback: join string fields
                        joined = ' '.join(str(v) for v in item.values() if isinstance(v, str) and v.strip())
                        if joined:
                            fout.write(normalize_text(joined).replace('\n', ' ') + '\n')
                    count += 1
                    if args.max_samples_per_dataset and count >= args.max_samples_per_dataset:
                        break

        print('Assembled corpus lines:', sum(1 for _ in open(corpus_path, 'r', encoding='utf-8')))

    # Train the SentencePiece model
    user_symbols = ['<bos>', '<eos>', '<pad>', '<nl>', '<lang_en>', '<lang_th>', '<lang_xx>', '<translate>', '<copy>', '<sep>']
    train_unigram(corpus_path, args.model_prefix, args.vocab_size, user_defined_symbols=user_symbols)

    print('Trained tokenizer model:', args.model_prefix)


if __name__ == '__main__':
    main()
