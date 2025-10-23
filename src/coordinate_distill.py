# src/coordinate_distill.py
import os
import argparse
import logging
import subprocess
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/coordinate_distill.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "generate_logits_grok.py")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "train_distill_amp_grok.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logits")
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")


def main(args):
    logging.info("🚀 开始协调生成分片、训练、删除循环...")
    logging.info(
        f"全量样本: {args.max_samples}, 分片大小: {args.shard_size}, 总分片数: {(args.max_samples + args.shard_size - 1) // args.shard_size}")

    for shard_idx in range((args.max_samples + args.shard_size - 1) // args.shard_size):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, args.max_samples)
        logging.info(f"分片 {shard_idx}: {start} 到 {end} 条")

        # 生成分片
        subprocess.call([
            "python", GENERATE_SCRIPT,
            "--dataset_path", args.dataset_path,
            "--batch_size", str(args.batch_size),
            "--max_seq_len", str(args.max_seq_len),
            "--max_samples", str(args.shard_size),
            "--start_from", str(start),
            "--device", args.device,
            "--compile" if args.compile else "",
            "--int8" if args.int8 else "",
            "--use_api" if args.use_api else "",
        ])
        logging.info(f"✅ 分片 {shard_idx} 生成完成")

        # 训练分片
        subprocess.call([
            "python", TRAIN_SCRIPT,
            "--teacher_logits_dir", OUTPUT_DIR,
            "--batch_size", str(args.batch_size),
            "--use_jsonl", "True" if args.use_api else "False",
            "--device", args.device,
            "--compile" if args.compile else "",
            "--max_samples_per_task", str(args.shard_size)
        ])
        logging.info(f"✅ 分片 {shard_idx} 训练完成，权重保存到 {OUTPUT_MODEL_DIR}")

        # 删除分片数据
        for file in [
            os.path.join(OUTPUT_DIR,
                         f"zh_to_en_shard_{shard_idx}.jsonl" if args.use_api else f"zh_to_en_shard_{shard_idx}.pt"),
            os.path.join(OUTPUT_DIR,
                         f"en_to_zh_shard_{shard_idx}.jsonl" if args.use_api else f"en_to_zh_shard_{shard_idx}.pt")
        ]:
            if os.path.exists(file):
                os.remove(file)
                logging.info(f"🗑️ 删除分片文件: {file}")

    logging.info("🎉 全量蒸馏完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="协调分片生成、训练、删除")
    parser.add_argument("--dataset_path", type=str, default="data/raw/wmt19_zh_en", help="WMT19路径")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--max_seq_len", type=int, default=64, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=26000000, help="全量样本数")
    parser.add_argument("--shard_size", type=int, default=100000, help="每片样本数")
    parser.add_argument("--compile", action="store_true", help="使用torch.compile")
    parser.add_argument("--int8", action="store_true", help="使用INT8量化")
    parser.add_argument("--use_api", action="store", help="使用DashScope API")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    main(args)
