# src/coordinate_distill.py (修正版)
import os
import argparse
import logging
import subprocess
import torch

# ===========================
# 日志配置
# ===========================
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
    total_shards = (args.max_samples + args.shard_size - 1) // args.shard_size
    logging.info(f"🚀 开始协调式蒸馏训练，共 {total_shards} 个分片")
    logging.info(f"全量样本: {args.max_samples:,}  分片大小: {args.shard_size:,}")

    for shard_idx in range(total_shards):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, args.max_samples)
        logging.info(f"\n{'='*70}\n🧩 分片 {shard_idx} [{start:,} - {end:,})\n{'='*70}")

        # --------------------------------------------------------
        # Step 1: 生成 Teacher Logits
        # --------------------------------------------------------
        cmd_gen = [
            "python", GENERATE_SCRIPT,
            "--dataset_path", args.dataset_path,
            "--batch_size", str(args.batch_size),
            "--max_seq_len", str(args.max_seq_len),
            "--max_samples", str(args.max_samples),
            "--start_from", str(start),
            "--shard_size", str(args.shard_size),
            "--device", args.device,
            "--shard_idx", str(shard_idx),
        ]
        if args.compile:
            cmd_gen.append("--compile")
        if args.int8:
            cmd_gen.append("--int8")
        if args.simulate_quant_noise:
            cmd_gen.append("--simulate_quant_noise")
            cmd_gen += ["--noise_std", str(args.noise_std)]

        logging.info(f"⚙️ 执行命令: {' '.join(cmd_gen)}")
        ret = subprocess.call(cmd_gen)
        if ret != 0:
            logging.error(f"❌ 分片 {shard_idx} Teacher Logits 生成失败，跳过。")
            continue
        logging.info(f"✅ 分片 {shard_idx} logits 生成完成")

        # --------------------------------------------------------
        # Step 2: 蒸馏训练学生模型
        # --------------------------------------------------------
        cmd_train = [
            "python", TRAIN_SCRIPT,
            "--teacher_logits_dir", OUTPUT_DIR,
            "--output_model_dir", OUTPUT_MODEL_DIR,
            "--batch_size", str(args.batch_size),
            "--max_samples_per_task", str(args.shard_size),
            "--device", args.device,
            "--shard_idx", str(shard_idx),
        ]
        if args.compile:
            cmd_train.append("--compile")

        logging.info(f"⚙️ 执行命令: {' '.join(cmd_train)}")
        ret = subprocess.call(cmd_train)
        if ret != 0:
            logging.error(f"❌ 分片 {shard_idx} 学生模型训练失败，跳过。")
            continue
        logging.info(f"✅ 分片 {shard_idx} 蒸馏训练完成")

        # --------------------------------------------------------
        # Step 3: 清理分片文件
        # --------------------------------------------------------
        for direction in ["zh_to_en", "en_to_zh"]:
            pt_file = os.path.join(OUTPUT_DIR, f"{direction}_shard_{shard_idx}.pt")
            if os.path.exists(pt_file):
                os.remove(pt_file)
                logging.info(f"🗑️ 删除分片文件: {pt_file}")

    logging.info("\n🎉 全量蒸馏训练完成！所有分片已处理。")
    logging.info(f"📦 最终学生模型目录: {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="协调分片生成 + 训练 + 清理 (GPU 优化版)")
    parser.add_argument("--dataset_path", type=str, default="data/raw/wmt19_zh_en", help="WMT19 数据路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小（GPU 可设大一点）")
    parser.add_argument("--max_seq_len", type=int, default=64, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=26000000, help="全量样本数")
    parser.add_argument("--shard_size", type=int, default=100000, help="每个分片的样本数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile")
    parser.add_argument("--int8", action="store_true", help="INT8 量化教师模型")
    parser.add_argument("--simulate_quant_noise", action="store_true", help="模拟量化噪声增强学生鲁棒性")
    parser.add_argument("--noise_std", type=float, default=0.01, help="模拟量化噪声标准差")
    args = parser.parse_args()

    main(args)