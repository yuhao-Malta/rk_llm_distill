import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.tiny_transformer import TinyTransformer

model = TinyTransformer(vocab_size=151936, max_seq_len=64)
try:
    model.load_state_dict(torch.load("outputs/models/student_model_amp_shard_0_best.pth"))
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")