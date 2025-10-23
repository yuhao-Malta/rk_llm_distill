# models/tiny_transformer.py
import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.quantization import QuantStub, DeQuantStub

class TinyTransformer(nn.Module):
    """
    轻量级 Transformer 学生模型（174M 参数，优化后目标：~40M参数，适配1126B NPU）
    支持多任务学习（中→英、英→中） + attention_mask来处理序列填充
    模型使用QWen tokenizer（从本地路径加载，以支持离线使用）
    基于PyTorch的nn.TransformerEncoder构建编码器部分
    端侧优化：Pre-norm、FP16/INT8、序列长度 64、低 batch 推理
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, max_seq_len=64, share_weights=True, vocab_size=None):
        """
        :param d_model: 隐藏维度（512→128，减小参数量）
        :param nhead: 注意力头数（8→4，适配 d_model）
        :param num_layers: Transformer 层数（6→2，减少参数）
        :param max_seq_len: 最大序列长度（64，适合短序列）
        :param share_weights: 是否共享 embed 和 lm_head 权重（减参数）
        :param vocab_size: 词汇表大小，从 config.json 读取
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.share_weights = share_weights

        # ✅ 指定本地 tokenizer 路径（相对于项目根目录），计算项目根目录并拼接本地路径
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")

        # ✅ 检查路径是否存在
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"❌ 本地 tokenizer 路径不存在: {TOKENIZER_PATH}")

        # ✅ 使用AutoTokenizer.from_pretrained从本地路径加载 tokenizer。与QWen教师模型共享tokenizer，便于蒸馏时token对齐
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_PATH, local_files_only=True, trust_remote_code=True
            )
            print("✅ 成功离线加载本地 Qwen Tokenizer！")
        except Exception as e:
            print(f"❌ 离线加载本地 tokenizer 失败: {e}")
            raise RuntimeError("无法加载本地 Qwen Tokenizer，请检查文件是否完整！")

        # 从 config.json 读取 vocab_size
        config_path = os.path.join(TOKENIZER_PATH, "config.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.vocab_size = config.get("vocab_size", len(self.tokenizer))
            print(f"✅ 使用 config.json 的词汇表大小: {self.vocab_size}")
        except Exception as e:
            print(f"⚠️ 读取 config.json 失败: {e}，使用 tokenizer 词汇表大小: {len(self.tokenizer)}")
            self.vocab_size = len(self.tokenizer)
        print(f"✅ Qwen Tokenizer 词汇表大小: {self.vocab_size}")  # 应输出 151963 或类似

        # ✅ 使用真实词汇表大小初始化 Embedding，输入token到d_model维嵌入
        self.embed = nn.Embedding(self.vocab_size, d_model)  # 参数量：151643*512=77.6M 参数
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        # 任务标识嵌入（0=中→英, 1=英→中）
        self.task_embed = nn.Embedding(2, d_model)  # 2*512=1K 参数

        # 输出头
        if share_weights:
            self.lm_head = self.embed
        else:
            self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,  # 2048→512，减少 FFN 参数
            dropout=0.1,
            batch_first=True,
            norm_first=True  # 端侧优化：Pre-norm提高训练稳定性和加速NPU推理（~10%延迟减少）
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

        # 量化支持
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.init_weights()

    def init_weights(self):
        """
        初始化：normal(std=0.02) for embed, lm_head（若不共享）, pos_embed
        端侧兼容：确保量化（INT8）后权重分布稳定
        """
        nn.init.normal_(self.embed.weight, std=0.02)
        if not self.share_weights:
            nn.init.normal_(self.lm_head.weight, std=0.02)  # 独立初始化lm_head
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.task_embed.weight, std=0.02)

    def forward(self, input_ids, task_id=None, attention_mask=None):
        # 确保 input_ids 是 LongTensor
        input_ids = input_ids.to(dtype=torch.long)
        # 量化输入（仅在推理时生效，训练时 QuantStub 是空操作）
        x = self.quant(input_ids)
        x = self.embed(x) * (self.d_model ** 0.5)
        seq_len = x.size(1)
        # ✅ 动态截断：如果 seq_len 超过 max_seq_len，则截断
        if seq_len > self.max_seq_len:
            print(f"⚠️  输入序列过长 ({seq_len} > {self.max_seq_len})，自动截断！")
            x = x[:, :self.max_seq_len, :]  # 截断输入
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]  # 同步截断 attention_mask
            seq_len = self.max_seq_len
        # 2.位置编码广播。截取位置编码并广播加到x
        x = x + self.pos_embed[:, :seq_len, :]  # 广播到 (batch, seq_len, d_model)

        # 3. 如果指定了任务，添加任务嵌入
        if task_id is not None:
            # 确保 task_id 是 LongTensor 并在正确设备上
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.long, device=input_ids.device)
            if task_id.dim() == 1:
                task_emb = self.task_embed(task_id).unsqueeze(1)  # (batch, 1, d_model)
                x = x + task_emb  # 广播到所有 token（相当于在序列开头注入任务信息）

        # 4. 注意力验码处理。如果提供了 attention_mask，生成 Transformer 需要的 key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            # TransformerEncoder 需要的是 key_padding_mask: (batch, seq_len)
            # True 表示要 mask 掉的位置，符合TransformerEncoder要求
            key_padding_mask = (attention_mask == 0)

        # 5. Transformer 编码.多头自注意力+FFN，处理序列依赖(确保LayerNorm输入和输出参数为float32)
        x = x.to(torch.float32)  # TransformerEncoder要求.强制float32，兼容CPU LayerNorm
        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask  # 👈 传递 key_padding_mask
        )
        # 解量化并生成 logits
        x = self.dequant(x)
        logits = self.lm_head(x)
        # 返回字典，便于后继损失计算，如CrossEntropyLoss
        return {"logits": logits}

    def num_parameters(self, verbose=False):
        """
        参数量计算，支持共享/不共享embed和lm_head权重
        修复Embed显示0.0M问题
        """
        total = 0
        breakdown = {
            'Embed' + (' (shared with LM Head)' if self.share_weights else ''): 0,
            'LM Head' + (' (shared)' if self.share_weights else ''): 0,
            'Encoder': 0,
            'Task Embed': 0,
            'Pos Embed': 0
        }

        for name, p in self.named_parameters():
            if p.requires_grad:
                param_count = p.numel()
                total += param_count
                if name.startswith('embed.'):
                    breakdown['Embed' + (' (shared with LM Head)' if self.share_weights else '')] = param_count
                elif name.startswith('lm_head.') and not self.share_weights:
                    breakdown['LM Head'] = param_count
                elif 'encoder' in name:
                    breakdown['Encoder'] += param_count
                elif name.startswith('task_embed.'):
                    breakdown['Task Embed'] = param_count
                elif name.startswith('pos_embed'):
                    breakdown['Pos Embed'] = param_count

        if verbose:
            print("| Component | Parameters (M) |")
            print("|-----------|----------------|")
            for comp, params in breakdown.items():
                print(f"| {comp} | {params / 1e6:.1f} |")
            print(f"| **Total** | **{total / 1e6:.1f}** |")

        return total


# -----------------------------
# 测试代码（验证模型是否可运行）
# -----------------------------
if __name__ == "__main__":
    # 测试不共享权重（174.2M）
    model = TinyTransformer(share_weights=False)
    print(f"✅ 模型参数量（不共享权重）: {model.num_parameters(verbose=True) / 1e6:.1f}M")

    input_ids = torch.randint(0, 32000, (2, 32))  # LongTensor
    attention_mask = torch.ones_like(input_ids, dtype=torch.float32)  # float32 for mask
    task_id = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():  # 端侧推理模拟
        outputs = model(input_ids, task_id=task_id, attention_mask=attention_mask)
    print(f"✅ 前向计算通过！Logits形状: {outputs['logits'].shape}")

    # 测试共享权重（96.6M）
    model_shared = TinyTransformer(share_weights=True)
    print(f"✅ 模型参数量（共享权重）: {model_shared.num_parameters(verbose=True) / 1e6:.1f}M")
