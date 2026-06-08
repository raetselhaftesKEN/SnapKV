from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    use_fast=False,
    force_download=True,
    resume_download=False,
)
print("ok")