#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 RKNN 量化校准数据集
从 teacher_logits 目录或训练样本中随机抽取 200 条输入样本
输出:
  - calibration_inputs.npy  (tokenized 输入)
  - calibration_texts.txt   (原始文本)
"""

import os
import torch
import numpy as np
import random
import logging
from config.config import TEACHER_LOGITS_DIR, ModelConfig
from models.tiny_seq2seq_transformer import TinySeq2SeqTransformer as TinyTransformer


# ========================
# 日志配置
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ========================
# 主函数
# ========================
def generate_calibration_dataset(
    teacher_logits_dir=TEACHER_LOGITS_DIR,
    output_dir="calibration_dataset",
    max_samples=200,
    max_seq_len=64
):
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"📦 从 {teacher_logits_dir} 加载蒸馏样本 ...")
    all_files = [
        f for f in os.listdir(teacher_logits_dir)
        if f.endswith(".pt")
    ]
    if not all_files:
        raise FileNotFoundError(f"❌ 未找到 teacher_logits 文件于 {teacher_logits_dir}")

    model = TinyTransformer(
        vocab_size=ModelConfig.VOCAB_SIZE,
        max_seq_len=max_seq_len,
        **ModelConfig.CURRENT_CONFIG
    )

    # 用于存储样本
    all_inputs = []
    all_texts = []

    for f in all_files:
        data_path = os.path.join(teacher_logits_dir, f)
        data = torch.load(data_path)
        logging.info(f"✅ 加载 {f}, 样本数: {len(data)}")

        for sample in data:
            if "src_input_ids" in sample:
                all_inputs.append(sample["src_input_ids"][:max_seq_len].unsqueeze(0))
            if "src_text" in sample:
                all_texts.append(sample["src_text"])

            if len(all_inputs) >= max_samples:
                break
        if len(all_inputs) >= max_samples:
            break

    if not all_inputs:
        raise ValueError("❌ 未提取到任何输入样本，请检查数据格式。")

    # 拼接为单个 tensor
    calib_tensor = torch.cat(all_inputs, dim=0)
    np.save(os.path.join(output_dir, "calibration_inputs.npy"), calib_tensor.numpy())
    logging.info(f"💾 保存 calibration_inputs.npy, 形状: {calib_tensor.shape}")

    # 保存原始文本
    if all_texts:
        with open(os.path.join(output_dir, "calibration_texts.txt"), "w", encoding="utf-8") as ftxt:
            for t in all_texts[:max_samples]:
                ftxt.write(t.strip() + "\n")
        logging.info(f"💾 保存 calibration_texts.txt ({len(all_texts[:max_samples])} 条)")
    else:
        logging.warning("⚠️ 数据中未包含 src_text 字段，仅保存 tokenized 输入。")

    logging.info(f"🎉 校准数据生成完成，输出目录: {output_dir}")


if __name__ == "__main__":
    generate_calibration_dataset()