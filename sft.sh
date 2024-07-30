#!/bin/bash

deepspeed /home/jc/workspace/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /home/jc/workspace/LLaVA/scripts/zero3.json \
    --model_name_or_path /date/jc/models/MedLLMs/LLaVA-Med/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path /home/jc/workspace/MedLLMs/Eso-Llava/data/processed_data/endo/precancer/full_data.json \
    --image_folder /home/jc/workspace/MedLLMs/Eso-Llava/data/processed_data/endo/precancer/images \
    --vision_tower /date/jc/models/MedLLMs/LLaVA-Med/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/jc/workspace/MedLLMs/Eso-Llava/checkpoints/240730v1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
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
    --report_to wandb

