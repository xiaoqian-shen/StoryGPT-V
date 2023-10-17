
DATASET=$1
CKPT_PATH=$2
GILL_CKPT=$3

OUTPUT_DIR="outputs/"

rm -r ${OUTPUT_DIR}

accelerate launch \
    --mixed_precision=fp16 \
    eval_llm.py \
    --finetuned_model_path ${CKPT_PATH} \
    --dataset_name data/${DATASET} \
    --dataset ${DATASET} \
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
    --inference_steps ${inference_steps} \
    --start_merge_step 15 \
    --no_object_augmentation \
    --ref_image 'same' \
    --gill_ckpt ${GILL_CKPT} \
    --interleave \
    # --story_len 50 \
    # --display_cap

cd eval

python eval_all.py --gen_dir ${OUTPUT_DIR} --dataset ${DATASET}
python eval_cap.py --gen_dir ${OUTPUT_DIR} --dataset ${DATASET}