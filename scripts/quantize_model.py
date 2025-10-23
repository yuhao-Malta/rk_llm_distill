import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.quantization
from models.tiny_transformer import TinyTransformer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/quantize_model.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 加载模型
try:
    model = TinyTransformer(
        vocab_size=151936,
        max_seq_len=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        share_weights=True
    )
    model.load_state_dict(torch.load("outputs/models/student_model_amp_shard_0_best.pth", map_location="cpu"))
    logging.info("✅ 模型加载成功: outputs/models/student_model_amp_shard_0_best.pth")
except Exception as e:
    logging.error(f"❌ 模型加载失败: {e}")
    raise

# 设置量化配置
try:
    model.eval()
    if hasattr(torch.quantization, 'float_qparams_weight_only_qconfig'):
        logging.info("✅ 使用 float_qparams_weight_only_qconfig 量化 nn.Embedding")
        # 为 nn.Embedding 和 nn.Linear 指定量化配置
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                torch.nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig
            },
            dtype=torch.qint8
        )
    else:
        logging.warning("⚠️ float_qparams_weight_only_qconfig 不可用，跳过 nn.Embedding 量化")
        # 仅量化 nn.Linear
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    logging.info("✅ 模型量化成功")
except Exception as e:
    logging.error(f"❌ 模型量化失败: {e}")
    raise

# 保存量化模型
try:
    output_path = "outputs/models/student_model_int8.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    logging.info(f"✅ 量化模型已保存至: {output_path}")
except Exception as e:
    logging.error(f"❌ 保存量化模型失败: {e}")
    raise