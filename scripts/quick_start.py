#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 一键测试完整流程
用法: python scripts/quick_start.py --mode [test|small|full]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import logging
from config.config import get_config_summary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_command(cmd, description):
    """运行命令并显示进度"""
    logging.info(f"\n{'=' * 60}")
    logging.info(f"🚀 {description}")
    logging.info(f"{'=' * 60}")
    logging.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logging.info(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {description} 失败: {e}")
        return False


def test_mode():
    """测试模式: 验证基础功能"""
    logging.info("\n" + "=" * 60)
    logging.info("🧪 测试模式: 验证基础功能")
    logging.info("=" * 60)

    # 1. 显示配置
    get_config_summary()

    # 2. 运行单元测试
    if not run_command(
            ["python", "tests/test_model.py"],
            "单元测试"
    ):
        return False

    # 3. 测试模型初始化
    if not run_command(
            ["python", "models/tiny_transformer.py"],
            "模型初始化测试"
    ):
        return False

    logging.info("\n✅ 测试模式完成！所有基础功能正常")
    return True


def small_mode():
    """小规模模式: 100 样本端到端测试"""
    logging.info("\n" + "=" * 60)
    logging.info("📦 小规模模式: 100 样本端到端测试")
    logging.info("=" * 60)

    # 1. 生成 teacher logits (100 样本)
    if not run_command(
            [
                "python", "scripts/generate_logits_grok.py",
                "--max_samples", "100",
                "--batch_size", "2",
                "--device", "cpu",
                "--int8"
            ],
            "生成 Teacher Logits (100 样本)"
    ):
        return False

    # 2. 训练学生模型 (100 样本)
    if not run_command(
            [
                "python", "src/train_distill_amp_grok.py",
                "--max_samples_per_task", "100",
                "--batch_size", "2",
                "--epochs", "2",
                "--device", "cpu"
            ],
            "训练学生模型 (100 样本)"
    ):
        return False

    # 3. 评估模型
    model_path = "outputs/models/student_model_amp_shard_0_best.pth"
    if os.path.exists(model_path):
        if not run_command(
                [
                    "python", "scripts/evaluate_model.py",
                    "--model_path", model_path,
                    "--max_samples", "50",
                    "--device", "cpu"
                ],
                "评估学生模型"
        ):
            return False
    else:
        logging.warning(f"⚠️ 模型文件未找到: {model_path}")

    logging.info("\n✅ 小规模模式完成！")
    logging.info("📊 下一步:")
    logging.info("  1. 检查日志文件: logs/")
    logging.info("  2. 查看模型权重: outputs/models/")
    logging.info("  3. 如需全量训练，运行: python scripts/quick_start.py --mode full")
    return True


def full_mode():
    """全量模式: 完整训练流程"""
    logging.info("\n" + "=" * 60)
    logging.info("🚀 全量模式: 完整训练流程")
    logging.info("=" * 60)
    logging.warning("⚠️ 全量训练需要大量时间和资源，建议在 GPU 服务器上运行")

    # 确认
    response = input("\n是否继续？(y/N): ")
    if response.lower() != 'y':
        logging.info("已取消")
        return False

    # ======== GPU 设定 ========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("⚠️ 未检测到 CUDA，将在 CPU 上运行（极慢！）")

    # RTX 5090 32GB 推荐设置：
    # - shard_size：10~20 万条一片
    # - batch_size：32（AMP模式下）
    # - compile：开启（torch.compile 可优化推理）
    # - simulate_quant_noise：True 提升学生鲁棒性

    max_samples = 10000
    shard_size = 1000  # 约130个分片，训练效率和文件管理更平衡
    batch_size = 2 if device == "cuda" else 4
    noise_std = 0.01  # 模拟量化噪声强度

    logging.info(
        f"💡 参数设定: max_samples={max_samples:,}, shard_size={shard_size:,}, batch_size={batch_size}, device={device}")
    batch_size = 16

    # ======== 启动协调式蒸馏训练 ========
    cmd = [
        "python", "src/coordinate_distill.py",
        "--dataset_path", "data/raw/wmt19_zh_en",
        "--max_samples", str(max_samples),
        "--shard_size", str(shard_size),
        "--batch_size", str(batch_size),
        "--device", device,
        "--compile",
        "--simulate_quant_noise",
        "--noise_std", str(noise_std)
    ]

    if not run_command(cmd, "协调式分片蒸馏训练"):
        return False

    # ======== 模型量化 (INT8) ========
    if not run_command(
            ["python", "scripts/quantize_model.py"],
            "模型量化 (INT8)"
    ):
        return False

    # ======== 评估阶段 ========
    final_models = {
        "best": "outputs/models/student_model_merged_best.pth",
        "int8": "outputs/models/student_model_int8.pth"
    }

    for model_type, path in final_models.items():
        if os.path.exists(path):
            run_command(
                [
                    "python", "scripts/evaluate_model.py",
                    "--model_path", path,
                    "--is_int8" if model_type == "int8" else "",
                    "--max_samples", "1000"
                ],
                f"评估 {model_type.upper()} 模型"
            )

    logging.info("\n✅ 全量蒸馏流程完成！")
    logging.info("📦 模型输出目录: outputs/models/")
    logging.info("  - FP32 模型: student_model_merged_best.pth")
    logging.info("  - INT8 模型: student_model_int8.pth")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速启动脚本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "small", "full"],
        default="test",
        help="运行模式: test=基础测试, small=小规模训练(100样本), full=全量训练"
    )
    args = parser.parse_args()

    # 显示欢迎信息
    print("\n" + "=" * 60)
    print("🎉 欢迎使用 RK_LLM_Distill 项目快速启动脚本")
    print("=" * 60)
    print(f"当前模式: {args.mode.upper()}")
    print("=" * 60)

    # 运行对应模式
    if args.mode == "test":
        success = test_mode()
    elif args.mode == "small":
        success = small_mode()
    elif args.mode == "full":
        success = full_mode()
    else:
        logging.error(f"未知模式: {args.mode}")
        success = False

    # 退出
    if success:
        print("\n✅ 任务完成！")
        sys.exit(0)
    else:
        print("\n❌ 任务失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
