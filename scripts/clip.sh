DATASET=$1

CKPT_PATH=$2
OUTPUT_DIR="outputs/"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=fp16 \
    models/clip_embs.py \
    --dataset_name data/${DATASET} \
    --dataset ${DATASET} \
    --finetuned_model_path ${CKPT_PATH} \
    --test_reference_folder data/${DEMO_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 30 \
    --start_merge_step 10 \
    --no_object_augmentation \
    --ref_image 'same' \