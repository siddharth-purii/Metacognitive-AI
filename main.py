import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datetime import datetime

MODEL_PATH = "C:/Projects/AI Models/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

def generate_response(prompt, max_new_tokens=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

def trap_transparency(user_input, initial_answer):
    prompt = f"User said: {user_input}\nAnswer given: {initial_answer}\nAnalyze transparency and assumptions in the answer:"
    return generate_response(prompt)

def trap_reasoning(initial_answer):
    prompt = f"Analyze reasoning quality and logic in this answer: {initial_answer}"
    return generate_response(prompt)

def trap_adaptability(user_input, initial_answer):
    prompt = f"Given user input: {user_input} and answer: {initial_answer}, suggest improvements or adaptations:"
    return generate_response(prompt)

def trap_perception(user_input):
    prompt = f"Analyze user's context, mood, or intent based on this input: {user_input}"
    return generate_response(prompt)

def generate_final_answer(user_input, trap_outputs):
    prompt = f"User input: {user_input}\nTransparency: {trap_outputs['transparency']}\nReasoning: {trap_outputs['reasoning']}\nAdaptability: {trap_outputs['adaptability']}\nPerception: {trap_outputs['perception']}\nUsing these insights, generate an improved answer:"
    return generate_response(prompt, max_new_tokens=256)

def run_trap_chatbot(user_input):
    initial_answer = generate_response(user_input)

    trap_outputs = {
        'transparency': trap_transparency(user_input, initial_answer),
        'reasoning': trap_reasoning(initial_answer),
        'adaptability': trap_adaptability(user_input, initial_answer),
        'perception': trap_perception(user_input),
    }

    final_answer = generate_final_answer(user_input, trap_outputs)

    return initial_answer, trap_outputs, final_answer

def save_log_to_excel(log_data, filename="trap_log.xlsx"):
    df = pd.DataFrame(log_data)
    df.to_excel(filename, index=False)

if __name__ == "__main__":
    log = []
    print("TRAP Chatbot started. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        print("\nGenerating initial answer...")
        initial, trap_outputs, final = run_trap_chatbot(user_input)

        print("\n--- Initial Answer ---")
        print(initial)
        print("\n--- TRAP Analysis ---")
        for k, v in trap_outputs.items():
            print(f"{k.capitalize()}:\n{v}\n")
        print("--- Final Improved Answer ---")
        print(final)

        log.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "initial_answer": initial,
            "transparency": trap_outputs['transparency'],
            "reasoning": trap_outputs['reasoning'],
            "adaptability": trap_outputs['adaptability'],
            "perception": trap_outputs['perception'],
            "final_answer": final
        })

    save_log_to_excel(log)
    print(f"\nLog saved to trap_log.xlsx")
