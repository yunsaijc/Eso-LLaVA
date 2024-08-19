#!/bin/bash

DATA_INDEX="endo/precancer"
# DATA_INDEX="CT/advanced"
date="240818"
version="e2"
base_model_path="/date/jc/models/MedLLMs/LLaVA-Med/"
# base_model="llava-med-v1.5-mistral-7b"
base_model="llava-v1.6-vicuna-13b"
data_dir="/date/jc/data/Eso-Llava/processed_data/${DATA_INDEX}/v8"
# data_path="/date/jc/data/Eso-Llava/processed_data/${DATA_INDEX}/v6/train.json"
image_folder="/date/jc/data/Eso-Llava/processed_data/${DATA_INDEX}/images"
output_dir="/date/jc/models/MedLLMs/LLaVA-Med/checkpoints/${date}${version}-lora-${base_model}"
epoch=10
batch_size=32

cd /home/jc/workspace/LLaVA/llava/
# conda activate llava

deepspeed /home/jc/workspace/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /home/jc/workspace/LLaVA/scripts/zero3.json \
    --model_name_or_path ${base_model_path}${base_model} \
    --version v1 \
    --data_path ${data_dir}/train.json \
    --image_folder ${image_folder} \
    --vision_tower ${base_model_path}clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 20 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb > /home/jc/workspace/MedLLMs/Eso-Llava/sft.log

# python /home/jc/workspace/LLaVA/scripts/merge_lora_weights.py \
#        --model-path ${output_dir} \
#        --model-base ${base_model_path}${base_model} \
#        --save-model-path "${base_model_path}/merged/${date}${version}-lora-${base_model}-merged"

#--validation_data_path ${data_dir}/val.json \

