import os
import time
import torch
from transformers import AutoTokenizer
from models_zoo.base_model import BaseModel
from models.tiny_transformer import TinyTransformer  # 👈 导入你原有的模型定义


class QwenDistillModel(BaseModel):
    def __init__(self, model_path="models/qwen_distill_80m.pth"):
        """
        初始化 Qwen 蒸馏模型
        :param model_path: 训练好的模型权重文件路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """加载 Qwen 蒸馏模型权重"""
        print(f"📥 加载 Qwen 蒸馏模型权重: {self.model_path}")

        # 1. 初始化模型架构（复用 tiny_transformer.py）
        self.model = TinyTransformer()  # 使用默认参数，或根据你的训练配置调整

        # 2. 加载训练好的权重
        if os.path.exists(self.model_path):
            # 加载权重（假设是 .pth 文件）
            checkpoint = torch.load(self.model_path, map_location='cpu')  # 先加载到 CPU，避免显存问题
            self.model.load_state_dict(checkpoint)
            print("✅ 模型权重加载成功！")
        else:
            print(f"⚠️  警告: 模型权重文件 {self.model_path} 不存在，将使用随机初始化权重！")
            # 你可以选择在这里抛出异常，或继续使用随机权重进行测试
            # raise FileNotFoundError(f"模型权重文件 {self.model_path} 不存在！")

        # 3. 设置为评估模式
        self.model.eval()

        # 4. 加载 Qwen Tokenizer（用于编码/解码）
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)
        print("✅ Qwen Tokenizer 加载成功！")

    def translate(self, text: str, src_lang: str = "zh", tgt_lang: str = "en") -> str:
        """翻译文本（中英互译）"""
        start_time = time.time()

        try:
            # 1. 编码输入文本
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

            # 2. 模型推理（前向计算）
            with torch.no_grad():  # 禁用梯度计算，加速推理
                outputs = self.model(inputs.input_ids)
                logits = outputs["logits"]

            # 3. 解码输出（贪心解码）
            # 选择概率最高的 token
            predicted_ids = torch.argmax(logits, dim=-1)
            # 将 token ID 转换为文本
            translated = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 4. 记录延迟
            self.latency = (time.time() - start_time) * 1000  # ms

            return translated

        except Exception as e:
            print(f"❌ 翻译出错: {str(e)}")
            return "翻译失败"

    def get_model_size(self) -> str:
        """返回模型大小（参数量）"""
        param_size = sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024  # MB
        return f"{param_size:.1f}MB"

    def get_latency(self) -> float:
        """返回推理延迟"""
        return getattr(self, 'latency', 0.0)