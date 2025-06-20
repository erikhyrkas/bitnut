import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BitNetForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_PATH = "./bitnut-small"  # pretrained base model path
DATA_DIR = "./bitnut_finetune"  # directory containing finetuning .jsonl files
MAX_LENGTH = 512
OUTPUT_DIR = "./bitnut-small-finetuned"  # result path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = BitNetForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)


def load_bitnut_data(data_dir):
    samples = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            with open(os.path.join(data_dir, filename), "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = f"{obj['prompt']}\n{obj['response']}"
                        samples.append({"text": text})
                    except Exception:
                        continue
    return Dataset.from_list(samples)


dataset = load_bitnut_data(DATA_DIR)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    save_steps=500,
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
