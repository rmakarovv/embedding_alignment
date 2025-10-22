python /root/embedding_alignment/train_kv_prefix.py \
    --train_csv russian_dialogues_embeddings/train.csv \
    --output_dir outputs_kv_prefix \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --emb_dim 768 \
    --hidden_dim 2048 \
    --num_kv_tokens 16 \
    --epochs 15 \
    --lr 0.00001

python /root/embedding_alignment/eval_kv_prefix.py \
    --test_csv russian_dialogues_embeddings/test.csv \
    --decoder_name Qwen/Qwen2.5-3B-Instruct \
    --projector_path outputs_kv_prefix/kv_projector_final.pt \
    --num_samples 10 \
    --max_new_tokens 256 \
    --do_sample
