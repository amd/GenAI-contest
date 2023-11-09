import torch
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

default_dataset_name = 'nisaar/Articles_Constitution_3300_Instruction_Set'

def add_parser_argument(parser):
    parser.add_argument(
        "--model-id", type=str, help="The model card id, finetuning will be on default model list if the option is not specified", required=True)
    parser.add_argument(
        "--use-bnb", action="store_true", help="Use bitsandbytes library to perform model quantizations")
    parser.add_argument(
        "--bnb-load-bit", type=int, default=4, choices=[4], help="Number of loading bits for target model")
    parser.add_argument(
        "--inference", action="store_true", help="Perform inference only")
    parser.add_argument(
        "--training-steps", type=int, default=20, help="Training steps")
    parser.add_argument(
        "--finetune-model-save-path", type=str, default='outputs', help="Model path for saving")
    parser.add_argument(
        "--prompt", type=str, default='What is large language model?', help="Input sentense to model for generation")
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Output sentense length")


def prepare_for_finetuning(args):

    model_id = args.model_id
    dataset = load_dataset(default_dataset_name, split="train")
    dataset = dataset.map(lambda example: {'text': example['prompt'] + example['output']})

    bnb_config = None
    if args.use_bnb:
        if args.bnb_load_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    #trainer arguments
    if args.training_steps < 1:
        print("Bad number of training steps")
        exit(1)
    
    training_arguments = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=4,
        save_steps=1,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=args.training_steps,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
    )

    max_seq_length = 2048

    peft_config = None
    if args.use_bnb:
        print("use bnb")
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )
        
    return trainer

def inference(args):

    device = torch.device("cuda:0")
        
    model = AutoModelForCausalLM.from_pretrained(args.finetune_model_save_path)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = args.prompt
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens)

    torch.cuda.synchronize()

    outputs = tokenizer.batch_decode(output_ids)

    print(outputs)
    
    
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
        model_to_save.save_pretrained(args.finetune_model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_argument(parser)
    args = parser.parse_args()
    print(args)
    main(args)



