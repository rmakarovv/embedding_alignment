# requirements
python -m pip install -r ./ALGEN/requirements.txt
python -m pip install sacrebleu

# training parameters
mdl_nm="google/mt5-base"
out_dir="./res"
max_len=512
data_fldr="../russian_dialogues_embeddings"
lang="ru"
train_size=5000
val_size=10
batch_size=8
lr=5e-5
wd=1e-4
num_epochs=5
wandb_run_nm="decoder_finetuning"
training_mode=""

mkdir -p "$out_dir"

args=(
    --model_name "$mdl_nm"
    --output_dir "$out_dir"
    --max_length "$max_len"
    --data_folder "$data_fldr"
    --lang "$lang"
    --train_samples "$train_size"
    --val_samples "$val_size"
    --batch_size "$batch_size"
    --learning_rate "$lr"
    --weight_decay "$wd"
    --num_epochs "$num_epochs"
    --wandb_run_name "$wandb_run_nm"
)

if [ -n "${training_mode}" ]; then
    args+=( --training_mode "$training_mode" )
fi

python /root/embedding_alignment/ALGEN/src/exp.py \
    "${args[@]}"
