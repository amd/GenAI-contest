# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import torch
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--finetuned_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to finetuned model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--fine_tuning_methods",
        type=int,
        default=None,
        required=True,
        help="Finetuned methods: 1. baseline 2. min-SNR 3. LoRA",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="a flying dragon",
        help="prompts for testing",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.finetuned_model_name_or_path is not None:
        if args.fine_tuning_methods == 1: # baseline
            model_path = args.finetuned_model_name_or_path
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            pipe.to("cuda")
            image = pipe(prompt=args.prompts).images[0]
            image.save(str(args.prompts).replace(" ", "_") + "_pokemon_baseline.png")
        elif args.fine_tuning_methods == 2: # min-SNR
            model_path = args.finetuned_model_name_or_path
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            pipe.to("cuda")
            image = pipe(prompt=args.prompts).images[0]
            image.save(str(args.prompts).replace(" ", "_") + "_pokemon_minSNR.png")
        elif args.fine_tuning_methods == 3: # LoRA
            model_path = args.pretrained_model_name_or_path
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            pipe.unet.load_attn_procs(args.finetuned_model_name_or_path)
            pipe.to("cuda")
            image = pipe(prompt=args.prompts).images[0]
            image.save(str(args.prompts).replace(" ", "_") + "_pokemon_LoRA.png")

if __name__ == "__main__":
    main()
