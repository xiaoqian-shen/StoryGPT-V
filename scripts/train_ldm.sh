DATASET=$1

current_month=$(date +%m)
current_day=$(date +%d)
current_hour=$(date +%H)
run_time=${current_month}-${current_day}-${current_hour}
export WANDB_NAME=${run_time}-${DATASET}-min1_mask3_text5
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0"

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

MODEL=stable-diffusion-v1-5

accelerate launch \
    --main_process_port ${PORT} \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 2 \
    --multi_gpu \
    models/train.py \
    --dataset_name data/${DATASET} \
    --dataset ${DATASET} \
    --logging_dir logs/${MODEL}/${DATASET}/${WANDB_NAME} \
    --output_dir checkpoints/${MODEL}/${DATASET}/${WANDB_NAME} \
    --num_train_epochs 300 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --checkpointing_steps 2000 \
    --mixed_precision bf16 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 2000 \
    --disable_flashattention \
    --resume_from_checkpoint latest \
    --report_to wandb \
    --enable_reg_loss \
    --train_image_encoder \
    --image_encoder_trainable_layers 2 \
    --text_only_prob 0.5 \
    --min_num_objects 1 \
    --mask_loss \
    --mask_loss_prob 0.3 \
    
    
    
