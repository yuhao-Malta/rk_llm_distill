# scripts/evaluate_model.py
import os
import sys
import gc
import time
import torch
import psutil
import logging
import sacrebleu
from tabulate import tabulate
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tiny_transformer import TinyTransformer
from config.config import (
    ModelConfig, EvalConfig, LogConfig,
    MODEL_PATH, RAW_DATA_PATH
)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format=LogConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LogConfig.EVALUATE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ==================== 辅助函数 ====================
def get_memory_usage():
    """获取当前进程内存占用 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def load_test_dataset(test_data_path, tasks=None, max_samples=100):
    """加载测试数据集 (支持中→英、英→中)"""
    tasks = tasks or EvalConfig.TASKS
    datasets = {}

    try:
        dataset = load_dataset("parquet", data_files={"test": test_data_path})["test"]
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logging.info(f"✅ 加载测试数据集: {test_data_path} (样本数: {len(dataset)})")

        if "translation" in dataset.column_names:
            for task in tasks:
                src_lang, tgt_lang = task.split('_to_')
                task_dataset = [
                    {"translation": {src_lang: item["translation"][src_lang], tgt_lang: item["translation"][tgt_lang]}}
                    for item in dataset
                    if src_lang in item["translation"] and tgt_lang in item["translation"]
                ]
                datasets[task] = task_dataset
                logging.info(f"  - {task}: {len(task_dataset)} 条")
        else:
            raise ValueError(f"数据集格式未知: {dataset.column_names}")

    except Exception as e:
        logging.error(f"❌ 加载数据集失败: {e}")
        raise

    return datasets


@torch.no_grad()
def compute_bleu(model, tokenizer, dataset, task="zh_to_en", device="cpu", max_seq_len=64, half=False):
    """计算 BLEU 分数"""
    model.eval()
    refs, hyps = [], []
    task_id = 0 if task == "zh_to_en" else 1

    for idx, item in enumerate(dataset):
        src_lang, tgt_lang = task.split('_to_')
        src_text = item["translation"][src_lang]
        ref_text = item["translation"][tgt_lang]

        try:
            inputs = tokenizer(
                src_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=True
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
            task_ids = torch.tensor([task_id], dtype=torch.long, device=device)

            outputs = model(input_ids, task_id=task_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            if half:
                logits = logits.float()  # 避免半精度CPU计算报错

            hyp_text = tokenizer.decode(
                logits.argmax(dim=-1)[0],
                skip_special_tokens=True
            )

            refs.append([ref_text])
            hyps.append(hyp_text)

            # 每 10 条释放内存
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache() if device == "cuda" else None
                gc.collect()

        except Exception as e:
            logging.error(f"❌ 翻译失败 ({task}, 文本: {src_text[:40]}...): {e}")
            continue

    try:
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        return bleu.score
    except Exception as e:
        logging.error(f"❌ 计算 BLEU 失败: {e}")
        return 0.0


def evaluate_student_model(model_path, test_data_path, device="cpu", max_samples=50, half=False):
    """主评估入口函数"""
    logging.info("=" * 60)
    logging.info("🔍 开始学生模型评估")
    logging.info("=" * 60)

    # 1️⃣ 加载模型
    logging.info(f"📦 加载学生模型: {model_path}")
    device = torch.device(device)
    model = TinyTransformer(
        vocab_size=ModelConfig.VOCAB_SIZE,
        max_seq_len=ModelConfig.MAX_SEQ_LEN,
        **ModelConfig.CURRENT_CONFIG
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    if half:
        model = model.half()
        logging.info("⚙️ 使用半精度推理模式 (float16)")

    logging.info(f"🧠 模型加载完成，占用内存: {get_memory_usage():.2f} MB")

    # 2️⃣ 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logging.info("✅ Tokenizer 加载成功")

    # 3️⃣ 加载测试集
    datasets = load_test_dataset(test_data_path, max_samples=max_samples)
    results_table = []

    # 4️⃣ 循环任务评估
    for task, data in datasets.items():
        logging.info(f"🧪 开始评估任务: {task}")
        start_time = time.time()
        bleu = compute_bleu(model, tokenizer, data, task, device, half=half)
        elapsed = time.time() - start_time
        mem = get_memory_usage()

        results_table.append([task, f"{bleu:.2f}", f"{elapsed:.2f}s", f"{mem:.2f}MB"])
        logging.info(f"✅ {task} BLEU={bleu:.2f}, ⏱️ 耗时={elapsed:.2f}s, 内存={mem:.2f}MB")

        del data
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # 5️⃣ 汇总打印表格
    logging.info("=" * 60)
    logging.info("📊 任务结果汇总")
    logging.info("\n" + tabulate(results_table, headers=["任务", "BLEU", "耗时", "内存占用"], tablefmt="fancy_grid"))
    avg_bleu = sum(float(row[1]) for row in results_table) / len(results_table)
    logging.info(f"🎯 平均 BLEU: {avg_bleu:.2f}")
    logging.info(f"🧠 最终内存占用: {get_memory_usage():.2f} MB")
    logging.info("✅ 评估完成！")
    logging.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估学生模型（支持中英互译）")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--is_int8", action="store_true", help="是否 INT8 模型")
    parser.add_argument("--tasks", type=str, nargs='+', default=EvalConfig.TASKS, help="任务列表")
    parser.add_argument("--max_samples", type=int, default=EvalConfig.MAX_EVAL_SAMPLES, help="每任务测试样本数")
    parser.add_argument("--test_data_path", type=str, default=EvalConfig.TEST_DATA_PATH, help="测试数据路径")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--half", action="store_true", help="是否启用半精度推理模式")
    args = parser.parse_args()

    evaluate_student_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        device=args.device,
        max_samples=args.max_samples,
        half=args.half
    )