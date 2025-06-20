import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load from your local fine-tuned BitNut directory
MODEL_DIR = "./bitnut-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("🐾 BitNut is ready. Ask me anything. Ctrl+C to exit.")

try:
    while True:
        user_input = input("\nYou: ")

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = model.generate(
                **generate_inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract response (skip prompt)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"BitNut: {response}")

except KeyboardInterrupt:
    print("\n👋 Goodbye. Stay vigilant — the squirrels are watching.")
