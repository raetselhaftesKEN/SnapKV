from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=False)
for i in range(1000):
    tok("How is the ground truth for fake news established?", return_tensors="pt")
print("ok")