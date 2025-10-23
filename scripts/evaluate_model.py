# scripts/evaluate_model.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import psutil
import time
import logging
import sacrebleu
from transformers import AutoTokenizer
from datasets import load_dataset
from models.tiny_transformer import TinyTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/evaluate_model.log", encoding='utf-8'), logging.StreamHandler()]
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")
WMT19_TEST_PATH = os.path.join(PROJECT_ROOT, "data/raw/wmt19_zh_en/validation/validation-00000-of-00001.parquet")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    logging.info("✅ 成功加载 tokenizer")
except Exception as e:
    logging.error(f"❌ 加载 tokenizer 失败: {e}")
    raise

def load_test_dataset(tasks=["zh_to_en", "en_to_zh"], max_samples=100):
    datasets = {}
    try:
        dataset = load_dataset("parquet", data_files={"test": WMT19_TEST_PATH})["test"]
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logging.info(f"✅ 加载 WMT19 验证数据集，样本数: {len(dataset)}")
        logging.info(f"数据集列名: {dataset.column_names}")
        logging.info(f"首个样本: {dataset[0]}")
        if "translation" in dataset.column_names:
            for task in tasks:
                src_lang, tgt_lang = task.split('_to_')
                task_dataset = [
                    {"translation": {src_lang: item["translation"][src_lang], tgt_lang: item["translation"][tgt_lang]}}
                    for item in dataset if src_lang in item["translation"] and tgt_lang in item["translation"]
                ]
                datasets[task] = task_dataset
                logging.info(f"✅ 构造任务数据集 ({task})，样本数: {len(task_dataset)}")
        else:
            raise ValueError(f"数据集格式未知：列名 {dataset.column_names}")
    except Exception as e:
        logging.error(f"❌ 加载数据集失败: {e}")
        raise
    return datasets

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def compute_bleu(model, dataset, task="zh_to_en", device="cpu", max_seq_len=64):
    model.eval()
    refs, hyps = [], []
    task_id = 0 if task == "zh_to_en" else 1
    for item in dataset:
        src_lang, tgt_lang = task.split('_to_')
        src_text = item["translation"][src_lang]
        ref_text = item["translation"][tgt_lang]
        try:
            inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None).to(device) if "attention_mask" in inputs else None
            task_ids = torch.full_like(input_ids, task_id, dtype=torch.long).to(device)
            with torch.no_grad():
                outputs = model(input_ids, task_id=task_ids, attention_mask=attention_mask)
            hyp_text = tokenizer.decode(outputs["logits"].argmax(dim=-1)[0], skip_special_tokens=True)
            refs.append([ref_text])
            hyps.append(hyp_text)
        except Exception as e:
            logging.error(f"❌ 翻译失败 ({task}, 文本: {src_text}): {e}")
            continue
    try:
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        return bleu.score
    except Exception as e:
        logging.error(f"❌ 计算 BLEU 失败: {e}")
        return 0.0

def measure_inference_latency(model, dataset, task="zh_to_en", device="cpu", max_seq_len=64, num_samples=100):
    model.eval()
    latencies = []
    task_id = 0 if task == "zh_to_en" else 1
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        src_lang = task.split('_to_')[0]
        src_text = item["translation"][src_lang]
        try:
            inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None).to(device) if "attention_mask" in inputs else None
            task_ids = torch.full_like(input_ids, task_id, dtype=torch.long).to(device)
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids, task_id=task_ids, attention_mask=attention_mask)
            latencies.append(time.time() - start_time)
        except Exception as e:
            logging.error(f"❌ 推理延迟测量失败 ({task}, 文本: {src_text}): {e}")
            continue
    if not latencies:
        logging.error(f"❌ 无有效推理延迟数据 ({task})")
        return float('inf')
    return sum(latencies) / len(latencies) * 1000

def main(model_path, is_int8=False, tasks=["zh_to_en", "en_to_zh"], max_samples=100):
    try:
        model = TinyTransformer(vocab_size=151936, max_seq_len=64, d_model=128, nhead=4, num_layers=2, share_weights=True).to("cpu")
        if is_int8:
            model = torch.quantization.quantize_dynamic(model, {nn.Embedding, nn.Linear}, dtype=torch.qint8)
            logging.info("✅ 应用动态量化")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=not is_int8)
        model.eval()
        logging.info(f"✅ 加载 {'INT8' if is_int8 else 'FP32'} 模型")
    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        raise
    datasets = load_test_dataset(tasks, max_samples)
    mem_usage = get_memory_usage()
    logging.info(f"内存占用: {mem_usage:.2f} MB")
    for task in tasks:
        logging.info(f"评估任务: {task}")
        bleu_score = compute_bleu(model, datasets[task], task=task, device="cpu", max_seq_len=64)
        logging.info(f"BLEU 分数 ({task}): {bleu_score:.2f}")
        avg_latency = measure_inference_latency(model, datasets[task], task=task, device="cpu", max_seq_len=64, num_samples=min(100, len(datasets[task])))
        logging.info(f"平均推理延迟 ({task}): {avg_latency:.2f} ms (100 条样本)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估学生模型（支持中英互译）")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--is_int8", action="store_true", help="是否 INT8 模型")
    parser.add_argument("--tasks", type=str, nargs='+', default=["zh_to_en", "en_to_zh"], help="任务方向")
    parser.add_argument("--max_samples", type=int, default=100, help="每任务测试样本数")
    args = parser.parse_args()
    main(args.model_path, args.is_int8, args.tasks, args.max_samples)

