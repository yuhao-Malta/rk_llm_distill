#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型量化脚本 (增强版)
====================================
支持命令行参数 + 输出对比信息报告

示例：
    python scripts/quantize_model.py \
        --input_model outputs/models/student_model_final_merged.pth \
        --output_model outputs/models/student_model_int8.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.quantization
from models.tiny_transformer import TinyTransformer
import logging
from datetime import datetime

# =====================
# 日志配置
# =====================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/quantize_model.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =====================
# 参数解析
# =====================
parser = argparse.ArgumentParser(description="动态量化 Transformer 学生模型")
parser.add_argument("--input_model", type=str, default="outputs/models/student_model_final_merged.pth",
                    help="待量化的模型路径")
parser.add_argument("--output_model", type=str, default="outputs/models/student_model_int8.pth",
                    help="保存量化模型的输出路径")
parser.add_argument("--report_path", type=str, default="logs/quantization_report.txt",
                    help="量化报告输出路径")
args = parser.parse_args()

# =====================
# 加载模型
# =====================
try:
    model = TinyTransformer(
        vocab_size=151936,
        max_seq_len=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        share_weights=True
    )

    logging.info(f"📦 正在加载模型: {args.input_model}")
    state_dict = torch.load(args.input_model, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    logging.info(f"✅ 模型加载成功: {args.input_model}")

except Exception as e:
    logging.error(f"❌ 模型加载失败: {e}")
    raise

# =====================
# 量化
# =====================
try:
    model.eval()
    if hasattr(torch.quantization, 'float_qparams_weight_only_qconfig'):
        logging.info("✅ 使用 float_qparams_weight_only_qconfig 量化 nn.Embedding 与 nn.Linear")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                torch.nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig
            },
            dtype=torch.qint8
        )
    else:
        logging.warning("⚠️ float_qparams_weight_only_qconfig 不可用，仅量化 Linear 层")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    logging.info("✅ 模型量化成功")
except Exception as e:
    logging.error(f"❌ 模型量化失败: {e}")
    raise

# =====================
# 保存完整量化模型
# =====================
try:
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(quantized_model, args.output_model)   # ✅ 保存完整模型对象
    logging.info(f"✅ 量化模型已完整保存至: {args.output_model}")
except Exception as e:
    logging.error(f"❌ 保存量化模型失败: {e}")
    raise

# =====================
# 📊 生成对比报告
# =====================
try:
    def sizeof(file_path):
        return os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0

    orig_size = sizeof(args.input_model)
    quant_size = sizeof(args.output_model)
    compression_ratio = orig_size / quant_size if quant_size > 0 else 0

    orig_param_count = len(state_dict.keys())
    quant_param_count = len(quantized_model.state_dict().keys())

    report_lines = [
        "=" * 70,
        f"🧮 模型量化对比报告",
        "=" * 70,
        f"🕒 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"📥 原始模型: {args.input_model}",
        f"📦 量化模型: {args.output_model}",
        "",
        f"🔢 参数数量: {orig_param_count:,} → {quant_param_count:,}",
        f"💾 模型体积: {orig_size:.2f} MB → {quant_size:.2f} MB",
        f"📉 压缩比: {compression_ratio:.2f}x",
        "",
        "✅ 量化类型: 动态量化 (Dynamic Quantization)",
        "✅ 涉及层: nn.Linear + nn.Embedding (如可用)",
        "=" * 70
    ]

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 同步输出到日志
    logging.info("\n" + "\n".join(report_lines))
    logging.info(f"📄 量化报告已保存至: {args.report_path}")

except Exception as e:
    logging.error(f"❌ 生成量化报告失败: {e}")
    raise