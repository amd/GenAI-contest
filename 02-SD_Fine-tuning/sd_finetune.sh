#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export SD_FT_OPTIONS_COMMON="\
    --mixed_precision="fp16" \
    --dataset_name=$DATASET_NAME \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --max_train_steps=833 \
    --max_grad_norm=1"

echo "[INFO] Stable diffusion finetuning on AMD Instinct GPUs" 
echo "Enter target finetuning methods between 0-3"
echo "1 baseline"
echo "2 min-SNR weighning"
echo "3 LoRA"
echo "0 run all of 1-3"
read n 

if [ $n -eq 1 ] || [ $n -eq 0 ];
then
echo "[INFO] baseline finetuning"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
echo "[INFO] Current Model: " $MODEL_NAME
echo "[INFO] Current Dataset: "$DATASET_NAME
accelerate launch train_text_to_image.py \
    $SD_FT_OPTIONS_COMMON \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --learning_rate=1e-05 \
    --use_ema \
    --snr_gamma 5.0 \
    --validation_prompt="flying dragon" --report_to="wandb" \
    --tracker_project_name="sdv1.4-finetune" \
    --output_dir="sd-pokemon-model"  
fi

if [ $n -eq 2 ] || [ $n -eq 0 ];
then
echo "[INFO] min-SNR weighning"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
echo "[INFO] Current Model: " $MODEL_NAME
echo "[INFO] Current Dataset: "$DATASET_NAME
accelerate launch train_text_to_image.py \
    $SD_FT_OPTIONS_COMMON \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --learning_rate=1e-05 \
    --use_ema \
    --snr_gamma 5.0 \
    --validation_prompt="flying dragon" --report_to="wandb" \
    --tracker_project_name="sdv1.4-finetune" \
    --output_dir="sd-pokemon-model_minsnr"  
fi

if [ $n -eq 3 ] || [ $n -eq 0 ];
then
echo "[INFO] LoRA" 
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
echo "[INFO] Current Model: " $MODEL_NAME
echo "[INFO] Current Dataset: "$DATASET_NAME
accelerate launch train_text_to_image_lora.py \
    $SD_FT_OPTIONS_COMMON \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --learning_rate=1e-04 \
    --validation_prompt="flying dragon" --report_to="wandb" \
    --output_dir="sd-pokemon-model_lora"  
fi
