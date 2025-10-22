#!/bin/bash

python train_soft_tokens.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir outputs/no_lora_ru \
    --save_every 20

python eval_soft_tokens.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-1.5B-Instruct \
    --projector_path outputs/no_lora_ru/mlp_projector_final.pt \
    --output_log no_lora_15_ru.txt \
    --num_samples 10 \
    --do_sample


python train_prev.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir outputs/no_lora_ru_3b    \
    --lr 0.00005 \
    --epochs 10 \
    --save_every 100

python eval_soft_tokens.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --projector_path outputs/no_lora_ru_3b/mlp_projector_final.pt \
    --output_log no_lora_3b_ru.txt \
    --num_samples 10 \
    --do_sample


python train_prev.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir outputs/lora_ru \
    --save_every 20 \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru

python eval_soft_tokens.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-1.5B-Instruct \
    --projector_path outputs/lora_ru/mlp_projector_final.pt \
    --output_log lora_15_ru.txt \
    --num_samples 10 \
    --do_sample \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru


python train_soft_tokens.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir outputs/lora_ru_3b \
    --save_every 100 \
    --lr 0.00001 \
    --epochs 15 \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b

python eval_soft_tokens.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --projector_path outputs/lora_ru_3b/mlp_projector_final.pt \
    --output_log lora_3b_ru.txt \
    --num_samples 10 \
    --do_sample \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b


python train_prev.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir outputs/lora_ru_3b_5 \
    --save_every 100 \
    --lr 0.00005 \
    --epochs 15 \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b_5

python eval_prev.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --projector_path outputs/lora_ru_3b_5/mlp_projector_final.pt \
    --output_log lora_3b_ru_5.txt \
    --num_samples 10 \
    --do_sample \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b_5


python train_prev.py \
    --train_csv ../russian_dialogues_embeddings/train.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir outputs/lora_ru_3b_5_20 \
    --save_every 100 \
    --lr 0.00005 \
    --epochs 20 \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b_5_20

python eval_soft_tokens.py \
    --test_csv ../russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --projector_path outputs/lora_ru_3b_5_20/mlp_projector_final.pt \
    --output_log lora_3b_ru_5_20.txt \
    --num_samples 10 \
    --do_sample \
    --use_lora \
    --lora_adapter_dir outputs/lora_adapter_ru_3b_5_20
