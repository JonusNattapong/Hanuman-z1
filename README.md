# Hanuman-z1

set CUDA_VISIBLE_DEVICES=0 && python train.py --tokenizer ZombitX64/gpt-oss-mini-think --seq-len 512 --batch-size 8 --epochs 3 --lr 5e-5 --n-embd 512 --n-layer 6 --n-head 8 --output-dir ./out_run1 --model-type simple --use-safetensors --use-think-head --think-aux-coef 1.0