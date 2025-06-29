import os
import torch
from datasets import load_dataset
from transformers import (
    BitNetConfig,
    BitNetForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    print("=" * 70)
    print("BITNET 1.58 PRETRAINING")
    print("=" * 70)

    DATASET_SIZE = 5_000_000  # 5M samples = ~5B tokens (this data set has about 39b possible tokens)
    MAX_LENGTH = 1024  # Sequence length for training
    MAX_POSITION_EMBEDDINGS = 4096  # Maximum sequence length model can handle

    print(f"Dataset size: {DATASET_SIZE:,} samples")
    print(f"Training sequence length: {MAX_LENGTH}")
    print(f"Max model capacity: {MAX_POSITION_EMBEDDINGS} tokens")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./bitnut-tokenizer", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    print(f"\nCreating BitNet configuration...")
    config = BitNetConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,  # GQA
        intermediate_size=2304,  # 3x hidden_size
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        bit_width=1.58,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        hidden_act="relu2",
        tie_word_embeddings=False,
        use_cache=False,  # Disable during training
    )

    print("Creating BitNet model...")
    model = BitNetForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"BitNet model created: {total_params:.1f}M parameters")

    model = model.to("cuda")
    print("Model moved to GPU")

    print(f"\nLoading dataset ({DATASET_SIZE:,} samples)...")
    raw = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")
    split = raw["train"].select(range(DATASET_SIZE)).train_test_split(test_size=0.01, seed=42)

    print(f"Dataset loaded:")
    print(f"   Train: {len(split['train']):,} samples")
    print(f"   Eval: {len(split['test']):,} samples")

    def tokenize(batch):
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"]
        }

    print("Tokenizing dataset...")
    tokenized = split.map(
        tokenize,
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Tokenizing"
    )

    print("Dataset tokenized")

    output_dir = f"./bitnut-small"

    batch_size = 16
    grad_accum = 16
    # effective batch size = 256 (16x16)
    effective_batch_size = batch_size * grad_accum

    print(f"\nTraining configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")

    # In the interest of time to train, I'm only using 5 billion training tokens, which might be enough.
    # The paper this is based on uses A LOT more tokens for there 110 million parameter version. We have
    # more tokens available in our dataset, but if 5 billion is terrible, using 3 epochs might still
    # be good enough -- though, realistically, I think just taking 15 billion or 20 billion from the
    # original dataset would have been better. I just know that that's a month of training, and I'm
    # maybe being a little optimistic that I can stop at 5 billion.
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=3,
        save_total_limit=3,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,
        max_grad_norm=1.0,
        logging_dir="./logs",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    total_steps = len(tokenized["train"]) // effective_batch_size * 3  # 3 epochs
    print(f"   Total training steps: {total_steps:,}")
    print(f"   Estimated training time: {total_steps * 1.5 / 60:.1f} minutes")

    print("\nTesting forward pass...")
    sample = tokenized["train"][0]
    test_input = {
        'input_ids': torch.tensor([sample['input_ids'][:256]]).to("cuda"),
        'attention_mask': torch.tensor([sample['attention_mask'][:256]]).to("cuda")
    }

    with torch.no_grad():
        outputs = model(**test_input, labels=test_input['input_ids'])

    print(f"Forward pass successful!")
    print(f"   Initial loss: {outputs.loss.item():.4f}")
    print(f"   Output shape: {outputs.logits.shape}")

    print("\n" + "=" * 70)
    print("STARTING BITNET 1.58 PRETRAINING")
    print("=" * 70)
    print("Press Ctrl+C to stop training and save current checkpoint")

    try:
        trainer.train()

        print("\nTRAINING COMPLETED SUCCESSFULLY!")
        print("Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(f"{output_dir}/training_info.txt", "w") as f:
            f.write(f"BitNet 1.58 Model - small\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(tokenized['train']):,}\n")
            f.write(f"Training sequence length: {MAX_LENGTH}\n")
            f.write(f"Max position embeddings: {MAX_POSITION_EMBEDDINGS}\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Bit width: 1.58\n")
            f.write(f"Architecture: BitNet with ReLU² activation\n")

        print(f"Model saved to: {output_dir}")
        print(f"Training info saved to: {output_dir}/training_info.txt")

        print("\nTesting trained model...")
        test_text = "The future of artificial intelligence"
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_text}")
        print(f"Generated: {generated_text}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        model.save_pretrained(f"{output_dir}-interrupted")
        tokenizer.save_pretrained(f"{output_dir}-interrupted")
        print(f"Checkpoint saved to: {output_dir}-interrupted")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("BITNET 1.58 PRETRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()