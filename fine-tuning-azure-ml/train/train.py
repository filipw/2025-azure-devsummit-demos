import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def build_chat_texts(tokenizer, system_prompt: str, user_prompt: str, completion: str):
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_only_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_only_text + completion + (tokenizer.eos_token or "")
    return prompt_only_text, full_text

def make_processor(tokenizer, max_seq_length: int, system_prompt: str):
    def processor(example):
        prompt = example["prompt"]
        completion = example["completion"]
        prompt_only_text, full_text = build_chat_texts(tokenizer, system_prompt, prompt, completion)

        prompt_ids = tokenizer(
            prompt_only_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]

        src_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * src_len + full_ids[src_len:]

        return {
            "input_ids": full_ids,
            "labels": labels,
        }
    return processor

@dataclass
class SimpleCausalCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()

        max_len = batch["input_ids"].size(1)
        labels = []
        for f in features:
            lab = f["labels"]
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            labels.append(lab)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="ID of the model to fine-tune.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data (jsonl).")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data (jsonl).")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--save_merged_model", type=bool, default=True, help="Merge LoRA adapters and save the full model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.model_output, exist_ok=True)

    # 1) Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")

    # 2) Model/tokenizer
    model_id = args.model_id
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3) System prompt (strict schema)
    system_prompt = (
        "You are a PII extraction assistant.\n"
        "Extract PII and return ONLY a JSON array (no prose) of objects with keys:\n"
        "- entity: one of [ADDRESS, NAME, EMAIL_ADDRESS, PHONE_NUMBER]\n"
        "- value: the extracted string\n"
        "If none, return [].\n"
        "Do not include any keys other than entity and value. Do not add commentary."
    )

    # 4) Preprocess to build masked labels
    processor = make_processor(tokenizer, args.max_seq_length, system_prompt)
    print("Tokenizing and masking datasets...")
    train_dataset = train_dataset.map(processor, remove_columns=train_dataset.column_names, desc="Tokenizing train")
    eval_dataset = eval_dataset.map(processor, remove_columns=eval_dataset.column_names, desc="Tokenizing eval")

    # 5) LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "fc1", "fc2"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 6) Trainer
    collator = SimpleCausalCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_output, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # 7) Train
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 8) Evaluate
    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # 9) Save
    if args.save_merged_model:
        print("Merging adapters and saving the full model...")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.model_output, safe_serialization=True)
    else:
        trainer.model.save_pretrained(args.model_output)
    tokenizer.save_pretrained(args.model_output)
    print(f"Model saved to: {args.model_output}")

if __name__ == "__main__":
    main()