import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import json


def create_custom_tokenizer(
        dataset_name="HuggingFaceTB/smollm-corpus",
        dataset_config="cosmopedia-v2",
        vocab_size=32768,
        output_dir="./bitnut-tokenizer",
        num_samples=1_000_000,
):
    print("=" * 70)
    print("TRAINING CUSTOM 32K BPE TOKENIZER")
    print("=" * 70)
    print(f"Dataset: {dataset_name}/{dataset_config}")
    print(f"Training samples: {num_samples:,}")
    print(f"Target vocab size: {vocab_size:,}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    print(f"Extracting {num_samples:,} text samples...")

    def text_iterator():
        count = 0
        for sample in dataset:
            if count >= num_samples:
                break
            yield sample["text"]
            count += 1
            if count % 100000 == 0:
                print(f"   Processed {count:,} samples...")

    print("\nInitializing BPE tokenizer...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<pad>",  # Padding token
            "<unk>",  # Unknown token
            "<bos>",  # Beginning of sequence
            "<eos>",  # End of sequence
            "<mask>",  # Mask token (which we probably don't need.)
        ],
        continuing_subword_prefix="",
        end_of_word_suffix="",
    )

    tokenizer.decoder = decoders.ByteLevel()

    print("\nTraining BPE tokenizer...")
    print("This may take several minutes...")

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    print("Tokenizer training completed!")

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Raw tokenizer saved to: {tokenizer_path}")

    print("\nCreating HuggingFace compatible tokenizer...")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    hf_tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace tokenizer saved to: {output_dir}")

    print("\nTesting tokenizer...")
    test_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are transforming technology.",
        "def hello_world():\n    print('Hello, World!')",
        "1 + 1 = 2, and 2 * 3 = 6"
    ]

    for text in test_texts:
        tokens = hf_tokenizer.encode(text)
        decoded = hf_tokenizer.decode(tokens)
        print(f"Text: {text}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Decoded: {decoded}")
        print()

    stats = {
        "tokenizer_type": "BPE",
        "vocab_size": len(tokenizer.get_vocab()),
        "training_samples": num_samples,
        "dataset": f"{dataset_name}/{dataset_config}",
        "special_tokens": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        },
        "token_ids": {
            "bos_token_id": hf_tokenizer.bos_token_id,
            "eos_token_id": hf_tokenizer.eos_token_id,
            "unk_token_id": hf_tokenizer.unk_token_id,
            "pad_token_id": hf_tokenizer.pad_token_id,
            "mask_token_id": hf_tokenizer.mask_token_id,
        }
    }

    with open(os.path.join(output_dir, "tokenizer_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Tokenizer Statistics:")
    print(f"   Actual vocab size: {len(tokenizer.get_vocab()):,}")
    print(f"   BOS token ID: {hf_tokenizer.bos_token_id}")
    print(f"   EOS token ID: {hf_tokenizer.eos_token_id}")
    print(f"   PAD token ID: {hf_tokenizer.pad_token_id}")
    print(f"   UNK token ID: {hf_tokenizer.unk_token_id}")

    print("\nTOKENIZER COMPARISON")
    print("=" * 50)

    tokenizers_to_compare = [
        ("Custom 32K", output_dir),
        ("GPT-2", "gpt2"),
        ("DialoGPT", "microsoft/DialoGPT-medium"),
    ]

    test_text = "The future of artificial intelligence and machine learning is very exciting! def train_model(): return 'success'"

    print(f"Test text: {test_text}")
    print()

    for name, model_name in tokenizers_to_compare:
        try:
            if name == "Custom 32K":
                tokenizer_comp = PreTrainedTokenizerFast.from_pretrained(model_name)
            else:
                tokenizer_comp = AutoTokenizer.from_pretrained(model_name)

            tokens = tokenizer_comp.encode(test_text)
            print(f"{name:12}: {len(tokens):3} tokens | vocab: {tokenizer_comp.vocab_size:,}")

        except Exception as e:
            print(f"{name:12}: Error loading - {e}")

    print("\nLower token count = better compression!")

    print("\n" + "=" * 70)
    print("CUSTOM TOKENIZER TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Tokenizer saved to: {output_dir}")
    print("You can now use this tokenizer in your BitNet training!")
    print(f"\nTo use in your training script:")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}', use_fast=True)")

    return hf_tokenizer


if __name__ == "__main__":
    try:
        # Train the 32K BPE tokenizer
        custom_tokenizer = create_custom_tokenizer()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()
