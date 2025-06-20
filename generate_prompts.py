import json
import random
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# if you don't have microsoft build tools in path (cl.exe) in your path
# you don't technically need it for inference.
torch._dynamo.config.suppress_errors = True

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T" # it's fast and efficient and fun to see how well it does.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BATCHES = 20
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 40
MAX_NEW_TOKENS = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

with open("topics.txt", "r") as f:
    topics = [line.strip() for line in f if line.strip()]

Path("human_prompts").mkdir(exist_ok=True)

global_prompt_set = set()

def parse_response_to_questions(output_text):
    lines = output_text.strip().split("\n")
    return [line.lstrip("1234567890.-• ").strip() for line in lines if "?" in line and len(line.strip()) > 10]

def generate_questions_for_topic(topic, num_batches=NUM_BATCHES):
    filepath = f"human_prompts/{topic.replace(' ', '_')}.jsonl"
    seen = set()
    written = 0

    for _ in range(num_batches):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Generate 10 realistic, unique questions a human might ask an AI about '{topic}'. Avoid jokes or fantasy. Just real questions."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_input = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **chat_input,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_output = tokenizer.decode(output[0][chat_input["input_ids"].shape[-1]:], skip_special_tokens=True)
        questions = parse_response_to_questions(raw_output)

        with open(filepath, "a") as f:
            for q in questions:
                if q and q not in seen and q not in global_prompt_set:
                    seen.add(q)
                    global_prompt_set.add(q)
                    f.write(json.dumps({"topic": topic, "prompt": q}) + "\n")
                    written += 1

        print(f"[{topic}] +{written} prompts so far")
        time.sleep(0.05)  # throttle to be kind to GPU

def main():
    for topic in topics:
        generate_questions_for_topic(topic)

if __name__ == "__main__":
    main()
