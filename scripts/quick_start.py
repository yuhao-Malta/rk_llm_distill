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
import time
import subprocess
import shutil
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
    """增强日志版：26M 全样本蒸馏 + 自动合并 + 量化 + 评估 + GPU监控"""
    logging.info("\n" + "=" * 80)
    logging.info("🚀 全量模式: 26M 样本蒸馏训练（增强日志版）")
    logging.info("=" * 80)
    logging.warning("⚠️ 请确保磁盘可用空间 ≥ 1TB，训练过程持续数小时")

    # ======== 用户确认 ========
    response = input("\n是否继续执行全量蒸馏训练？(y/N): ")
    if response.lower() != 'y':
        logging.info("❌ 用户取消全量训练")
        return False

    # ======== 环境设定 ========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("⚠️ 未检测到 CUDA，将在 CPU 上运行（极慢！）")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "full_mode_run.log")

    # ======== 参数配置 ========
    max_samples = 26_000_000
    shard_size = 100_000
    batch_size = 8
    noise_std = 0.01

    logging.info(f"💡 参数设定:")
    logging.info(f"   max_samples = {max_samples:,}")
    logging.info(f"   shard_size  = {shard_size:,}")
    logging.info(f"   batch_size  = {batch_size}")
    logging.info(f"   device      = {device}")
    logging.info(f"   noise_std   = {noise_std}")
    logging.info(f"   日志文件    = {log_file}")

    # ======== 辅助函数 ========
    def gpu_status():
        """读取当前 GPU 状态"""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                text=True
            ).strip().split("\n")[0]
            temp, util, mem_used, mem_total, power = map(float, out.split(", "))
            return f"GPU {util:.0f}% | Mem {mem_used:.0f}/{mem_total:.0f} MB | Temp {temp:.0f}°C | Power {power:.0f}W"
        except Exception:
            return "GPU 状态不可用"

    def log_and_time(cmd, desc):
        """执行命令并计时 + GPU监控"""
        logging.info(f"\n{'=' * 80}")
        logging.info(f"🚀 开始: {desc}")
        logging.info(f"{'=' * 80}")
        logging.info(f"命令: {' '.join(cmd)}")

        start_time = time.time()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {desc} =====\n命令: {' '.join(cmd)}\n")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            peak_mem = 0
            for line in proc.stdout:
                line_stripped = line.strip()
                if "CUDA out of memory" in line_stripped:
                    logging.error("💥 检测到 OOM 错误！")
                if "MiB" in line_stripped and "allocated" in line_stripped:
                    try:
                        mem_val = int(line_stripped.split("MiB")[0].split()[-1])
                        peak_mem = max(peak_mem, mem_val)
                    except:
                        pass
                if time.time() % 60 < 1:  # 每分钟记录一次GPU状态
                    logging.info("📊 GPU监控: " + gpu_status())
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line)
            proc.wait()
            end_time = time.time()
            elapsed_min = (end_time - start_time) / 60
            logging.info(f"✅ {desc} 完成，用时 {elapsed_min:.1f} 分钟")
            logging.info(f"📈 峰值显存约: {peak_mem} MiB")
            return True
        except Exception as e:
            logging.error(f"❌ {desc} 失败: {e}")
            return False

    # ======== Step 1: 协调式蒸馏训练 ========
    cmd_train = [
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
    if not log_and_time(cmd_train, "协调式分片蒸馏训练"):
        return False

    # ======== Step 2: 模型量化 (INT8) ========
    cmd_quant = [
        "python", "scripts/quantize_model.py",
        "--input_model", "outputs/models/student_model_final_merged.pth",
        "--output_model", "outputs/models/student_model_int8.pth",
        "--report_path", "logs/quantization_report.txt"
    ]
    if not log_and_time(cmd_quant, "模型量化 (INT8)"):
        return False

    # ======== Step 3: 模型评估 ========
    models_to_eval = {
        "FP32": "outputs/models/student_model_final_merged.pth",
        "INT8": "outputs/models/student_model_int8.pth"
    }

    for model_name, model_path in models_to_eval.items():
        if os.path.exists(model_path):
            cmd_eval = [
                "python", "scripts/evaluate_model.py",
                "--model_path", model_path,
                "--max_samples", "1000"
            ]
            if "INT8" in model_name:
                cmd_eval.append("--is_int8")
            log_and_time(cmd_eval, f"评估 {model_name} 模型")
        else:
            logging.warning(f"⚠️ 未找到模型: {model_path}")

    # ======== Step 4: 汇总报告 ========
    report_path = os.path.join(log_dir, "distill_run_report.txt")
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("=== RK-LLM Distillation 全流程报告 ===\n")
        rpt.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rpt.write(f"总样本数: {max_samples:,}\n")
        rpt.write(f"分片大小: {shard_size:,}\n")
        rpt.write(f"批量大小: {batch_size}\n")
        rpt.write(f"噪声强度: {noise_std}\n")
        rpt.write(f"设备: {device}\n")
        rpt.write(f"GPU 状态: {gpu_status()}\n")
        rpt.write(f"\n输出模型:\n")
        for model_name, model_path in models_to_eval.items():
            rpt.write(f"  - {model_name}: {model_path}\n")
        rpt.write("\n查看完整日志: logs/full_mode_run.log\n")

    logging.info("\n✅ 全流程完成！详细报告已保存:")
    logging.info(f"📄 {report_path}")
    logging.info("📊 日志文件:")
    logging.info(f"   {log_file}")
    logging.info("📦 模型输出目录: outputs/models/")

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
