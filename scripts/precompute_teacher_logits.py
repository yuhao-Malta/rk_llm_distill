# script/precompute_teacher_logits.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/precompute_logits.log", encoding='utf-8'), logging.StreamHandler()]
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")
TEACHER_LOGITS_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logits")
CACHE_DIR = os.path.join(TEACHER_LOGITS_DIR, "cache")

def precompute_teacher_logits(file_path, output_path, max_seq_len=64, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cpu", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()

    data = []
    logging.info(f"Processing {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            hyp_encoded = tokenizer(item["hyp"], return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
            with torch.no_grad():
                logits = model(input_ids=hyp_encoded.input_ids, attention_mask=hyp_encoded.attention_mask).logits
            data.append({
                "src": item["src"],
                "hyp": item["hyp"],
                "task_id": item["task_id"],
                "src_input_ids": hyp_encoded.input_ids.squeeze(0),
                "src_attention_mask": hyp_encoded.attention_mask.squeeze(0),
                "teacher_logits": logits.squeeze(0)
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    logging.info(f"Saved precomputed logits to {output_path}")

if __name__ == "__main__":
    for task in ["zh_to_en", "en_to_zh"]:
        input_path = os.path.join(TEACHER_LOGITS_DIR, f"{task}_shard_0.jsonl")
        output_path = os.path.join(CACHE_DIR, f"{task}_shard_0.pt")
        precompute_teacher_logits(input_path, output_path, max_seq_len=64, max_samples=1000)
