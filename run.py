import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./bitnut-small-finetuned"

tokenizer = AutoTokenizer.from_pretrained("./bitnut-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("🐾 BitNut is ready to chase squirrels. Ask your questions. CTRL+C to stop.")

try:
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        prompt = f"human: {user_input}\nbitnut:"

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

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"BitNut: {response}")

except KeyboardInterrupt:
    print("\n🦴 BitNut will dream about squirrels.")
