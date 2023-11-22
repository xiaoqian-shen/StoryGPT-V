current_month=$(date +%m)
current_day=$(date +%d)
current_hour=$(date +%H)
run_time="${current_month}-${current_day}-${current_hour}"

DATASET=$1
llm_model=$2

python -u train_gill.py \
    --exp-name ${run_time}_${DATASET_NAME}_interleave_fuse --log-base-dir='logs/gill/' \
    --precision='bf16'  --print-freq=100 --batch-size=64  --val-batch-size=64 \
    --dataset-dir data/${DATASET} \
    --dataset ${DATASET}\
    --wandb \
    --model-modes generation \
    --max-len 160 \
    --clip_emb_file clip_emb_img.pkl \
    --num-tokens 8 \
    --interleave \
    --llm_model ${llm_model} \
    
