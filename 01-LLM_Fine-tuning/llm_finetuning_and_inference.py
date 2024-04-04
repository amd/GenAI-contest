# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer

default_dataset_name = "mlabonne/guanaco-llama2-1k"

def add_parser_argument(parser):
    parser.add_argument(
        "--model-id", type = str, help = "model card id", required = True)
    parser.add_argument(
        "--inference", action = "store_true", help = "Perform inference only")


def prepare_for_finetuning(args):

    model_id = args.model_id
    dataset = load_dataset(default_dataset_name, split = "train")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        trust_remote_code = True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token

    training_arguments = TrainingArguments(
        output_dir = "./results",
        num_train_epochs = 1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        optim = "paged_adamw_32bit",
        save_steps = 50,
        logging_steps = 50,
        learning_rate = 4e-5,
        weight_decay = 0.001,
        fp16=False,
        bf16=False,
        max_grad_norm = 0.3,
        max_steps = -1,
        warmup_ratio = 0.03,
        group_by_length = True,
        lr_scheduler_type = "constant",
        report_to = "tensorboard"
    )

    peft_config = LoraConfig(
        lora_alpha = 16,
        lora_dropout = 0.1,
        r = 64,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config = peft_config,
        dataset_text_field = "text",
        tokenizer = tokenizer,
        args = training_arguments,
    )

    return trainer

def inference(args):

    device = torch.device("cuda:0")

    base_model_id = args.model_id
    new_model_id = './model-enhanced'

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code = True, padding_side = "left")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = "Write an introduction about deez nuts"
    input_ids = tokenizer(inputs, return_tensors = "pt").input_ids.to(device)
    max_length = 64
    # base model
    output_ids = base_model.generate(input_ids, max_length = max_length)
    output = tokenizer.decode(output_ids[0], skip_special_tokens = True)
    print(f"[Base model generation] {output} \n")

    # new model
    base_model.config.to_json_file(new_model_id + "/config.json")
    new_model = AutoModelForCausalLM.from_pretrained(new_model_id).eval().to(device)
    output_ids = new_model.generate(input_ids, max_length = max_length)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"[New model generation] {output} \n")


def merge_unload(args):

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        return_dict=True,
        torch_dtype=torch.float16,
    )

    new_model = PeftModel.from_pretrained(base_model, "./temp")
    new_model = new_model.merge_and_unload()
    new_model.save_pretrained("./model-enhanced")


def finetuning(trainer):

    trainer.train()

def main(args):

    if args.inference:
        inference(args)
    else:
        trainer = prepare_for_finetuning(args)

        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

        finetuning(trainer)

        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.save_pretrained("./temp")

        merge_unload(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_argument(parser)
    args = parser.parse_args()

    main(args)
