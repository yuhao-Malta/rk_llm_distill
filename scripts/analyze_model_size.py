# scripts/analyze_model_size.py
import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from colorama import Fore, Style

# 兼容导入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tiny_transformer import TinyTransformer
from config.config import ModelConfig

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


# ==================== 辅助函数 ====================
def get_file_size(path):
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0.0


def estimate_memory_usage(param_count, dtype="fp32"):
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}.get(dtype, 4)
    mem = param_count * bytes_per_param / 1024 / 1024
    return round(mem * 1.8, 2)


def count_parameters_by_module(model):
    stats = {}
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        stats[top] = stats.get(top, 0) + p.numel()
    return stats


def count_parameters_by_layer(model):
    stats = {}
    for name, p in model.named_parameters():
        if "encoder.layers" in name:
            layer_id = name.split('.')[2]
            stats[layer_id] = stats.get(layer_id, 0) + p.numel()
    return stats


def estimate_flops(cfg):
    """粗略估算 FLOPs"""
    d_model = cfg["d_model"]
    num_layers = cfg["num_layers"]
    seq_len = 64  # 以 ModelConfig.MAX_SEQ_LEN 为基准
    attn_flops = 4 * seq_len * d_model * d_model
    ffn_flops = 2 * seq_len * d_model * d_model * 2
    return (attn_flops + ffn_flops) * num_layers / 1e6  # MFLOPs


# ==================== 主分析逻辑 ====================
def analyze_model(model_path=None, breakdown_layer=False):
    cfg = ModelConfig.CURRENT_CONFIG
    vocab_size = ModelConfig.VOCAB_SIZE

    logging.info(f"📘 使用模型配置: {cfg}")
    logging.info(f"📗 词汇表大小: {vocab_size}")

    # 构建模型
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        share_weights=cfg.get("share_weights", True),
    )

    # 加载权重
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"✅ 成功加载权重文件: {model_path}")
        except Exception as e:
            logging.warning(f"⚠️ 无法加载权重文件 ({e})，仅进行结构分析。")

    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6
    file_size = get_file_size(model_path)

    # 模块分布
    sub_stats = count_parameters_by_module(model)
    sub_total = sum(sub_stats.values())
    sub_stats = {k: v / sub_total * 100 for k, v in sub_stats.items()}

    # 内存占用估算
    fp32_size = total_params * 4 / 1024 / 1024
    fp16_size = total_params * 2 / 1024 / 1024
    int8_size = total_params * 1 / 1024 / 1024
    mem_int8 = estimate_memory_usage(total_params, "int8")

    # FLOPs
    total_flops = estimate_flops(cfg)

    # ==================== 主表格 ====================
    table = PrettyTable()
    table.field_names = ["精度类型", "参数量 (M)", "文件大小 (MB)", "推理内存 (MB)"]
    table.add_row(["FP32", f"{total_params_m:.2f}", f"{fp32_size:.2f}", f"{estimate_memory_usage(total_params, 'fp32'):.2f}"])
    table.add_row(["FP16", f"{total_params_m:.2f}", f"{fp16_size:.2f}", f"{estimate_memory_usage(total_params, 'fp16'):.2f}"])
    table.add_row(["INT8", f"{total_params_m:.2f}", f"{int8_size:.2f}", f"{mem_int8:.2f}"])

    print("\n" + "=" * 65)
    print(Fore.CYAN + "📊 TinyTransformer 模型分析报告 (兼容版)" + Style.RESET_ALL)
    print("=" * 65)
    print(table)

    # 模块占比
    print("\n" + Fore.YELLOW + "🔍 模块参数分布：" + Style.RESET_ALL)
    detail = PrettyTable()
    detail.field_names = ["模块", "参数占比 (%)"]
    for name, pct in sorted(sub_stats.items(), key=lambda x: x[1], reverse=True):
        color = Fore.GREEN if "encoder" in name else Fore.CYAN
        detail.add_row([color + name + Style.RESET_ALL, f"{pct:.2f}"])
    print(detail)

    # 分层分析
    if breakdown_layer:
        layer_stats = count_parameters_by_layer(model)
        if layer_stats:
            print("\n" + Fore.MAGENTA + "📐 Encoder 层参数分析：" + Style.RESET_ALL)
            layer_table = PrettyTable()
            layer_table.field_names = ["层编号", "参数量 (K)", "占比 (%)"]
            total = sum(layer_stats.values())
            for lid, count in layer_stats.items():
                layer_table.add_row([lid, f"{count / 1e3:.1f}", f"{count / total * 100:.2f}"])
            print(layer_table)

            plt.figure(figsize=(8, 4))
            plt.bar(range(len(layer_stats)), [v / 1e6 for v in layer_stats.values()])
            plt.xlabel("Encoder Layer ID")
            plt.ylabel("Params (Million)")
            plt.title("TinyTransformer 每层参数分布")
            plt.tight_layout()
            plt.show()

    # ==================== 汇总 ====================
    print("\n" + "-" * 65)
    print(f"📁 权重文件: {model_path or '未提供'} ({file_size:.2f} MB)")
    print(f"⚙️ 总参数量: {total_params_m:.2f}M")
    print(f"🧮 估算 FLOPs: {total_flops:.1f} MFLOPs")
    print("-" * 65)

    print(Fore.GREEN + "💡 部署建议：" + Style.RESET_ALL)
    if mem_int8 < 50:
        print("✅ 适合 RV1126B NPU 实时部署（INT8）")
    elif mem_int8 < 150:
        print("⚠️ 可运行于 Jetson Nano / RK3588")
    else:
        print("❌ 模型过大，建议进一步蒸馏或剪枝")

    print(f"🏗️ 部署格式建议: ONNX INT8 / RKNN int8")
    print(f"📦 推理内存预估: {mem_int8:.2f} MB")
    print("=" * 65 + "\n")


# ==================== 主入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze TinyTransformer model size and structure (compatible version)")
    parser.add_argument("--model_path", type=str, default="outputs/models/student_model_amp_shard_0_best.pth")
    parser.add_argument("--breakdown-layer", action="store_true", help="Show per-layer breakdown")
    args = parser.parse_args()

    analyze_model(args.model_path, breakdown_layer=args.breakdown_layer)