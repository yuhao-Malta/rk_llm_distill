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
from config.config import (
    ModelConfig, EvalConfig, LogConfig,
    MODEL_PATH, RAW_DATA_PATH
)

# 配置日志
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
    """
    加载测试数据集 (支持自定义路径)

    :param test_data_path: 测试数据路径 (parquet文件)
    :param tasks: 任务列表 ["zh_to_en", "en_to_zh"]
    :param max_samples: 每个任务的最大样本数
    :return: {task: [{"translation": {"zh": ..., "en": ...}}]}
    """
    tasks = tasks or EvalConfig.TASKS
    datasets = {}

    try:
        # 加载 parquet 文件
        dataset = load_dataset("parquet", data_files={"test": test_data_path})["test"]
        dataset = dataset.select(range(min(max_samples, len(dataset))))

        logging.info(f"✅ 加载测试数据集: {test_data_path} (样本数: {len(dataset)})")

        # 按任务分组
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


def compute_bleu(model, tokenizer, dataset, task="zh_to_en", device="cpu", max_seq_len=64):
    """
    计算 BLEU 分数

    :param model: 学生模型
    :param tokenizer: Tokenizer
    :param dataset: 测试数据集
    :param task: 任务名称
    :param device: 计算设备
    :param max_seq_len: 最大序列长度
    :return: BLEU 分数
    """
    model.eval()
    refs, hyps = [], []
    task_id = 0 if task == "zh_to_en" else 1

    for item in dataset:
        src_lang, tgt_lang = task.split('_to_')
        src_text = item["translation"][src_lang]
        ref_text = item["translation"][tgt_lang]

        try:
            # 分词
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

            # 推理
            with torch.no_grad():
                outputs = model(input_ids, task_id=task_ids, attention_mask=attention_mask)

            # 解码
            hyp_text = tokenizer.decode(
                outputs["logits"].argmax(dim=-1)[0],
                skip_special_tokens=True
            )

            refs.append([ref_text])
            hyps.append(hyp_text)

        except Exception as e:
            logging.error(f"❌ 翻译失败 ({task}, 文本: {src_text[:50]}...): {e}")
            continue

    # 计算 BLEU
    try:
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        return bleu.score
    except Exception as e:
        logging.error(f"❌ 计算 BLEU 失败: {e}")
        return 0.0


def measure_inference_latency(model, tokenizer, dataset, task="zh_to_en", device="cpu", max_seq_len=64,
                              num_samples=100):
    """
    测量推理延迟

    :param model: 学生模型
    :param tokenizer: Tokenizer
    :param dataset: 测试数据集
    :param task: 任务名称
    :param device: 计算设备
    :param max_seq_len: 最大序列长度
    :param num_samples: 测试样本数
    :return: 平均延迟 (ms)
    """
    model.eval()
    latencies = []
    task_id = 0 if task == "zh_to_en" else 1

    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        src_lang = task.split('_to_')[0]
        src_text = item["translation"][src_lang]

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

            # 测量时间
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids, task_id=task_ids, attention_mask=attention_mask)
            latencies.append(time.time() - start_time)

        except Exception as e:
            logging.error(f"❌ 推理失败 ({task}, 文本: {src_text[:50]}...): {e}")
            continue

    if not latencies:
        logging.error(f"❌ 无有效推理数据 ({task})")
        return float('inf')

    return sum(latencies) / len(latencies) * 1000  # 转换为 ms


# ==================== 主评估函数 ====================
def main(
        model_path,
        is_int8=False,
        tasks=None,
        max_samples=None,
        test_data_path=None,
        device="cpu"
):
    """
    评估学生模型 (支持中英互译)

    :param model_path: 模型权重路径
    :param is_int8: 是否 INT8 模型
    :param tasks: 任务列表
    :param max_samples: 每任务测试样本数
    :param test_data_path: 测试数据路径 (可选)
    :param device: 计算设备
    """
    # 默认值
    tasks = tasks or EvalConfig.TASKS
    max_samples = max_samples or EvalConfig.MAX_EVAL_SAMPLES
    test_data_path = test_data_path or EvalConfig.TEST_DATA_PATH

    logging.info("=" * 60)
    logging.info("🔍 开始模型评估")
    logging.info("=" * 60)
    logging.info(f"📦 模型路径: {model_path}")
    logging.info(f"🧪 测试数据: {test_data_path}")
    logging.info(f"📊 任务: {tasks}")
    logging.info(f"💻 设备: {device}")

    # 1. 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        logging.info("✅ Tokenizer 加载成功")
    except Exception as e:
        logging.error(f"❌ Tokenizer 加载失败: {e}")
        raise

    # 2. 加载模型
    try:
        model = TinyTransformer(
            vocab_size=ModelConfig.VOCAB_SIZE,
            max_seq_len=ModelConfig.MAX_SEQ_LEN,
            **ModelConfig.CURRENT_CONFIG
        ).to(device)

        # INT8 量化
        if is_int8:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Embedding, nn.Linear}, dtype=torch.qint8
            )
            logging.info("✅ INT8 量化已应用")

        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=not is_int8)
        model.eval()

        logging.info(f"✅ 模型加载成功 ({'INT8' if is_int8 else 'FP32'})")
    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        raise

    # 3. 加载测试数据
    datasets = load_test_dataset(test_data_path, tasks, max_samples)

    # 4. 内存占用
    mem_usage = get_memory_usage()
    logging.info(f"📊 内存占用: {mem_usage:.2f} MB")

    # 5. 评估各任务
    results = {}
    for task in tasks:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"📝 评估任务: {task}")
        logging.info(f"{'=' * 60}")

        # BLEU 分数
        bleu_score = compute_bleu(
            model, tokenizer, datasets[task],
            task=task, device=device, max_seq_len=ModelConfig.MAX_SEQ_LEN
        )
        logging.info(f"  BLEU 分数: {bleu_score:.2f}")

        # 推理延迟
        avg_latency = measure_inference_latency(
            model, tokenizer, datasets[task],
            task=task, device=device, max_seq_len=ModelConfig.MAX_SEQ_LEN,
            num_samples=min(100, len(datasets[task]))
        )
        logging.info(f"  平均延迟: {avg_latency:.2f} ms")

        results[task] = {
            "bleu": bleu_score,
            "latency_ms": avg_latency
        }

    # 6. 性能总结
    logging.info(f"\n{'=' * 60}")
    logging.info("📊 评估总结")
    logging.info(f"{'=' * 60}")
    logging.info(f"内存占用: {mem_usage:.2f} MB (目标: <{EvalConfig.TARGET_MEMORY_MB} MB)")
    for task, metrics in results.items():
        logging.info(f"{task}:")
        logging.info(f"  - BLEU: {metrics['bleu']:.2f} (目标: >{EvalConfig.TARGET_BLEU})")
        logging.info(f"  - 延迟: {metrics['latency_ms']:.2f} ms (目标: <{EvalConfig.TARGET_LATENCY_MS} ms)")

    # 7. 性能达标检查
    all_pass = True
    if mem_usage > EvalConfig.TARGET_MEMORY_MB:
        logging.warning(f"⚠️ 内存超标: {mem_usage:.2f} MB > {EvalConfig.TARGET_MEMORY_MB} MB")
        all_pass = False

    for task, metrics in results.items():
        if metrics['bleu'] < EvalConfig.TARGET_BLEU:
            logging.warning(f"⚠️ {task} BLEU 未达标: {metrics['bleu']:.2f} < {EvalConfig.TARGET_BLEU}")
            all_pass = False
        if metrics['latency_ms'] > EvalConfig.TARGET_LATENCY_MS:
            logging.warning(f"⚠️ {task} 延迟超标: {metrics['latency_ms']:.2f} ms > {EvalConfig.TARGET_LATENCY_MS} ms")
            all_pass = False

    if all_pass:
        logging.info("\n✅ 所有性能指标达标！")
    else:
        logging.info("\n⚠️ 部分性能指标未达标，需要优化")

    return results


# ==================== 主程序 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估学生模型（支持中英互译）")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--is_int8", action="store_true", help="是否 INT8 模型")
    parser.add_argument("--tasks", type=str, nargs='+', default=EvalConfig.TASKS, help="任务列表")
    parser.add_argument("--max_samples", type=int, default=EvalConfig.MAX_EVAL_SAMPLES, help="每任务测试样本数")
    parser.add_argument("--test_data_path", type=str, default=EvalConfig.TEST_DATA_PATH, help="测试数据路径")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        is_int8=args.is_int8,
        tasks=args.tasks,
        max_samples=args.max_samples,
        test_data_path=args.test_data_path,
        device=args.device
    )
