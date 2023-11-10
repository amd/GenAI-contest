#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
echo "[INFO] Baseline Model: " $MODEL_NAME
echo "[INFO] Stable diffusion inference test on AMD Instinct GPUs" 
echo "Enter prompt"
read prompt 
echo "Enter target finetuning methods between 0-3"
echo "1 baseline"
echo "2 min-SNR weighning"
echo "3 LoRA"
echo "0 run all of 1-3"
read n 

if [ $n -eq 1 ] || [ $n -eq 0 ];
then
echo "[INFO] baseline finetuning"
python inference_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --finetuned_model_name_or_path="sd-pokemon-model" \
    --fine_tuning_methods=1 \
    --prompts="$prompt"
fi

if [ $n -eq 2 ] || [ $n -eq 0 ];
then
echo "[INFO] min-SNR weighning"
python inference_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --finetuned_model_name_or_path="sd-pokemon-model_minsnr" \
    --fine_tuning_methods=2 \
    --prompts="$prompt"
fi

if [ $n -eq 3 ] || [ $n -eq 0 ];
then
echo "[INFO] LoRA" 
python inference_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --finetuned_model_name_or_path="sd-pokemon-model_lora" \
    --fine_tuning_methods=3 \
    --prompts="$prompt"
fi
