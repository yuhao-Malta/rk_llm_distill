#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_student_checkpoints.py
====================================
用于将多分片蒸馏生成的学生模型（*_best.pth）聚合成一个最终模型。

✅ 功能特性
- 自动扫描 outputs/models 下的所有 *_best.pth 文件
- 支持普通平均与加权平均融合
- 可选择 CPU / GPU 加载
- 输出 student_model_final_merged.pth
- 可选验证参数形状一致性

示例：
    python merge_student_checkpoints.py \
        --model_dir outputs/models \
        --output_path outputs/models/student_model_final_merged.pth \
        --device cpu \
        --mode mean

作者: Yuhao + GPT-5
日期: 2025-11-05
"""

import os
import glob
import torch
import argparse
import logging
from tqdm import tqdm

# =========================
# 🚀 参数定义
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple shard student checkpoints.")
    parser.add_argument("--model_dir", type=str, default="outputs/models",
                        help="目录路径，包含多个 *_best.pth 文件")
    parser.add_argument("--output_path", type=str, default="outputs/models/student_model_final_merged.pth",
                        help="合并后输出模型路径")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="加载时使用的设备")
    parser.add_argument("--mode", type=str, default="mean",
                        choices=["mean", "weighted"],
                        help="聚合模式：普通平均(mean) 或 加权平均(weighted)")
    parser.add_argument("--weights", type=float, nargs="*",
                        help="可选权重列表，对应每个模型（仅在 weighted 模式生效）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印模型文件，不实际合并")
    return parser.parse_args()

# =========================
# 🧠 日志配置
# =========================
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

# =========================
# 🔍 检查模型一致性
# =========================
def check_model_shapes(models):
    ref_keys = set(models[0].keys())
    for i, state_dict in enumerate(models[1:], start=1):
        if set(state_dict.keys()) != ref_keys:
            missing = ref_keys - set(state_dict.keys())
            extra = set(state_dict.keys()) - ref_keys
            raise ValueError(f"❌ 模型 {i} 参数键不一致。\n缺失: {missing}\n多余: {extra}")

# =========================
# ⚙️ 主聚合逻辑
# =========================
def merge_checkpoints(args):
    model_paths = sorted(glob.glob(os.path.join(args.model_dir, "*_best.pth")))
    if not model_paths:
        logging.error(f"❌ 未找到任何 *_best.pth 文件，请检查路径: {args.model_dir}")
        return

    logging.info(f"📦 共找到 {len(model_paths)} 个分片模型：")
    for i, path in enumerate(model_paths):
        logging.info(f"  [{i+1:02d}] {path}")

    if args.dry_run:
        logging.info("🟡 Dry-run 模式，仅列出模型，不进行合并。")
        return

    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logging.info(f"💻 使用设备: {device}")

    # 载入第一个模型
    logging.info(f"📥 加载模型: {model_paths[0]}")
    merged_state = torch.load(model_paths[0], map_location=device)
    for key in merged_state.keys():
        merged_state[key] = merged_state[key].float()

    # 检查权重长度
    if args.mode == "weighted":
        if not args.weights or len(args.weights) != len(model_paths):
            raise ValueError("❌ 加权模式需要指定与模型数量一致的 --weights 参数。")
        total_weight = sum(args.weights)
        normalized_weights = [w / total_weight for w in args.weights]
    else:
        normalized_weights = [1.0 / len(model_paths)] * len(model_paths)

    # 加载并累加后续模型
    for i, path in enumerate(tqdm(model_paths[1:], desc="🔄 聚合中")):
        state_dict = torch.load(path, map_location=device)
        check_model_shapes([merged_state, state_dict])

        weight = normalized_weights[i + 1] if args.mode == "weighted" else normalized_weights[i + 1]
        for key in merged_state.keys():
            merged_state[key] += state_dict[key].float() * weight

    # 保存聚合结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(merged_state, args.output_path)
    logging.info(f"✅ 聚合完成！输出文件: {args.output_path}")
    logging.info(f"📏 参数数量: {len(merged_state)}")

# =========================
# 🎯 主函数入口
# =========================
if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    merge_checkpoints(args)