import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.suppress_errors = True

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
# I'm going to force cpu because even though it is slower, it frees up my gpu for pretraining
# I figure that I might as well be generating finetuning data while pretraining is running.
DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 40
MAX_NEW_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  # when you batch and have a decoder-only models, this makes it likely it'll work

INPUT_DIR = "human_prompts"
OUTPUT_DIR = "bitnut_finetune"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

BITNUT_STYLE = (
    "You are BitNut, a very good golden retriever who believes squirrels are behind all major global events. "
    "You get distracted easily, talk about treats, tail-chasing, and squirrel surveillance, and speak with excited urgency. "
    "Respond to the following prompt in that voice. Keep responses under 200 words."
)


def process_batch(prompts_batch):
    if not prompts_batch:
        return []
    start_time = time.time()  # Add this line

    all_prompts = []
    for prompt_text in prompts_batch:
        messages = [
            {"role": "system", "content": BITNUT_STYLE},
            {"role": "user", "content": prompt_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    batch_inputs = tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    total_new_tokens = 0
    responses = []
    for i, output in enumerate(outputs):
        # Skip the input tokens for each sequence in the batch
        input_length = batch_inputs.input_ids[i].shape[0]
        response_tokens = output[input_length:]
        total_new_tokens += len(response_tokens)
        raw_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
        responses.append(raw_output.strip())

    tokens_per_second = total_new_tokens / generation_time if generation_time > 0 else 0
    print(f"  Generated {total_new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")

    return responses


def process_file(input_file):
    filename = Path(input_file).name
    output_file = Path(OUTPUT_DIR) / filename

    print(f"Processing {filename}")

    prompts = []
    with open(input_file, "r") as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
                prompts.append(item["prompt"])
            except Exception as e:
                print(f"Skipping invalid line: {e}")
                continue

    print(f"Loaded {len(prompts)} prompts")

    batch_size = 4
    processed = 0

    with open(output_file, "w") as outfile:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            print(f"Processing batch of {len(batch)} prompts...")

            responses = process_batch(batch)

            for prompt_text, response in zip(batch, responses):
                result = {
                    "prompt": f"human: {prompt_text}",
                    "response": f"bitnut: {response}"
                }
                outfile.write(json.dumps(result) + "\n")
                processed += 1

            print(f"[{filename}] Processed {processed}/{len(prompts)} prompts")

            # Minimal delay between batches
            if i + batch_size < len(prompts):
                time.sleep(0.05)


def main():
    input_files = list(Path(INPUT_DIR).glob("*.jsonl"))
    print(f"Found {len(input_files)} files to process")

    for input_file in input_files:
        process_file(input_file)
        print(f"Completed {input_file.name}")


if __name__ == "__main__":
    main()
