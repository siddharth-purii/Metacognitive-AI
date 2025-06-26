# Metacogntive AI Application - TRAP Chatbot

This is a Python-based chatbot application leveraging a causal language model to generate responses and perform multi-faceted analysis on them. The chatbot uses the **TRAP framework** (Transparency, Reasoning, Adaptability, Perception) to analyze and improve its own generated answers before presenting the final improved response.

---

## Features

- Generates an initial answer to a user's input using a pre-trained causal language model.
- Analyzes the initial answer on four key aspects:
  - **Transparency:** Checks clarity and assumptions.
  - **Reasoning:** Evaluates logic and quality of reasoning.
  - **Adaptability:** Suggests improvements or adaptations based on the input.
  - **Perception:** Attempts to understand the user's context, mood, or intent.
- Generates a final improved answer using insights from the TRAP analysis.
- Logs all conversations and analyses, saving them to an Excel file (`trap_log.xlsx`) upon exit.
- Automatically detects and uses GPU if available.

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers (`transformers` library)
- Pandas
- OpenPyXL (for Excel output)
- TinyLlama-1.1B-Chat-v1.0

---

## Code Overview
- Model loading and device setup: Loads tokenizer and model, moves model to GPU if available.
- generate_response(prompt, ...): Generates a text response from the model given a prompt.
- TRAP functions (trap_transparency, trap_reasoning, trap_adaptability, trap_perception): Generate prompts for each TRAP dimension and get model responses.
- generate_final_answer(...): Combines TRAP insights to produce an improved final answer.
- run_trap_chatbot(user_input): Runs full pipeline for a single user input.
- Logging and saving: Logs conversation data and saves it to Excel on exit.

---

## About This Implementation
A very basic implementation of the TRAP framework can be found here. This entry-level model is intended primarily as a proof of concept rather than a ready-for-production solution. It does not modify the underlying system architecture but functions as a wrapper built on top of the existing model.
