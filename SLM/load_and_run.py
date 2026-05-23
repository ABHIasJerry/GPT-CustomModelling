
####################### NOT IN USE #######################
# from transformers import pipeline
# pipe = pipeline("text-generation", model="./my_slm", tokenizer="./my_slm")
# print(pipe("The main goal of this project is", max_length=50))
##########################################################

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import warnings
warnings.filterwarnings("ignore") # Suppress all warnings
warnings.warn("This will not be shown", UserWarning)
print("[Warning Supressed] > Script continues without warning output")

# Load model and tokenizer
model_dir = "./my_slm"
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}\n")

# Generate text
prompt = input("Enter the prompt: ")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")