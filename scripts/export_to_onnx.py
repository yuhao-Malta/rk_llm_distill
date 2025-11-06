import sys

import torch
import os
from models.tiny_transformer import TinyTransformer

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def export_onnx(model_path="../outputs/models/student_model_amp_shard_0_best.pth",
                onnx_path="../outputs/models/student_model_amp_shard_0_best.onnx"):
    # ✅ 1. 加载 PyTorch 模型
    model = TinyTransformer()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # ✅ 2. 构造示例输入
    batch_size, seq_len = 2, 32
    dummy_input = torch.randint(0, 151936, (batch_size, seq_len))

    # ✅ 3. 导出 ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=13
    )

    print(f"✅ 模型已导出: {onnx_path}")

if __name__ == "__main__":
    export_onnx()