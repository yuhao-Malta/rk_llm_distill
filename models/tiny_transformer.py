# models/tiny_transformer.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class TinyTransformer(nn.Module):
    """
    轻量级 Transformer 学生模型（目标：50M~80M 参数）
    支持多任务学习（中→英、英→中） + attention_mask来处理序列填充
    模型使用QWen tokenizer（从本地路径加载，以支持离线使用）
    基于PyTorch的nn.TransformerEncoder构建编码器部分
    """

    def __init__(self, d_model=512, nhead=8, num_layers=6, max_seq_len=64):
        """
        :param d_model: 隐藏维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param max_seq_len: 最大序列长度
        """
        super().__init__()
        # self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ✅ 指定本地 tokenizer 路径（相对于项目根目录），计算项目根目录并拼接本地路径
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")

        # ✅ 检查路径是否存在
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"❌ 本地 tokenizer 路径不存在: {TOKENIZER_PATH}")

        # ✅ 使用AutoTokenizer.from_pretrained从本地路径加载 tokenizer。与QWen教师模型共享tokenizer，便于蒸馏时token对齐
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_PATH,  # 👈 关键修改：使用本地路径
                local_files_only=True,  # 👈 强制离线加载
                trust_remote_code=True
            )
            print("✅ 成功离线加载本地 Qwen Tokenizer！")
        except Exception as e:
            print(f"❌ 离线加载本地 tokenizer 失败: {e}")
            raise RuntimeError("无法加载本地 Qwen Tokenizer，请检查文件是否完整！")

        # ✅ 获取真实词汇表大小，用于后继Embedding初始化
        self.vocab_size = len(self.tokenizer)
        print(f"✅ Qwen Tokenizer 词汇表大小: {self.vocab_size}")  # 应输出 151643 或类似

        # ✅ 使用真实词汇表大小初始化 Embedding，输入token到d_model维嵌入
        self.embed = nn.Embedding(self.vocab_size, d_model)  # 参数量：151643*512=77.6M 参数
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # 0 参数（buffer）

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers)  # 每层约3.15M，6 层 Transformer 约 18.9M 参数（包括自注意力、FFN等）

        # 任务标识嵌入（0=中→英, 1=英→中）
        self.task_embed = nn.Embedding(2, d_model)  # 2*512=1K 参数

        # 输出头。从隐藏状态映射到词汇表
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)  # 77.6M 参数（可以与embed共享权重）

        self.init_weights()
        # 权重共享：LM Head 直接复用 Embed 的权重（形状相同，无需转置）
        # self.lm_head.weight = self.embed.weight  # 共享同一张量（非 Parameter 包装，避免多余 overhead）。可以缩小参量，但最好教师模型也共享权重

    def init_weights(self):
        """
        初始化权重。使用正泰分布（std=0.02）初始化embed、lm_head和pos_embed
        Xavier/Glorot初始化未用，适合Transformer
        """
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids, task_id=None, attention_mask=None):
        """
        前向传播。这是模型的核心逻辑，支持因果语言建模（Causal LM），但使用Encoder而非Decoder（无自回归mask，适合编码任务如翻译）
        :param input_ids: (batch_size, seq_len) token ID张量
        :param task_id: (batch_size,) 任务标识，用于区分翻译方向
        :param attention_mask: (batch_size, seq_len) 可选，用于屏蔽填充部分（1=关注，0=掩码填充）
        :return: logits 用于下一个token的概率分布，形状为(batch_size, seq_len, vocab_size)，
        """
        # ✅ 调试：检查 input_ids 范围[0,vocab_size-1]
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        if max_id >= self.vocab_size or min_id < 0:
            raise ValueError(
                f"❌ input_ids 超出范围! 允许范围: [0, {self.vocab_size - 1}], 实际范围: [{min_id}, {max_id}]")

        # 1. 输入嵌入 + 位置编码
        x = self.embed(input_ids)  # (batch, seq, d_model)  #token嵌入
        seq_len = x.size(1)
        # ✅ 动态截断：如果 seq_len 超过 max_seq_len，则截断
        if seq_len > self.max_seq_len:
            print(f"⚠️  输入序列过长 ({seq_len} > {self.max_seq_len})，自动截断！")
            x = x[:, :self.max_seq_len, :]  # 截断输入
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]  # 同步截断 attention_mask
            seq_len = self.max_seq_len
        # 2.位置编码广播。截取位置编码并广播加到x
        pos_enc = self.pos_embed[:, :seq_len, :]  # (1, seq_len, d_model)
        x = x + pos_enc  # 广播到 (batch, seq_len, d_model)

        # 3. 注意力验码处理。如果提供了 attention_mask，生成 Transformer 需要的 key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            # TransformerEncoder 需要的是 key_padding_mask: (batch, seq_len)
            # True 表示要 mask 掉的位置，符合TransformerEncoder要求
            key_padding_mask = (attention_mask == 0)

        # 4. 如果指定了任务，添加任务嵌入
        if task_id is not None:
            # 确保 task_id 是 LongTensor 并在正确设备上
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.long, device=input_ids.device)
            if task_id.dim() == 1:
                task_emb = self.task_embed(task_id).unsqueeze(1)  # (batch, 1, d_model)
                x = x + task_emb  # 广播到所有 token（相当于在序列开头注入任务信息）

        # 5. Transformer 编码.多头自注意力+FFN，处理序列依赖
        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask  # 👈 传递 key_padding_mask
        )

        # 5. 预测 logits：线性投影到词汇表
        logits = self.lm_head(x)
        # 返回字典，便于后继损失计算，如CrossEntropyLoss
        return {"logits": logits}

    def num_parameters(self, verbose=False):
        """
        返回可训练参数量，自动处理权重共享（如 embed 和 lm_head tied 时不重复计算）
        :param verbose: 如果 True，打印各组件参数 breakdown
        """
        total = 0
        seen_data_ptr = set()  # 跟踪已计入的存储指针，避免重复
        breakdown = {}  # 用于 verbose 的组件统计

        for name, p in self.named_parameters():
            if p.requires_grad:
                data_ptr = p.data_ptr()
                if data_ptr not in seen_data_ptr:
                    seen_data_ptr.add(data_ptr)
                    total += p.numel()

                    # 分类 breakdown（可选，基于名称）
                    if 'embed' in name:
                        breakdown['Embed'] = p.numel()
                    elif 'lm_head' in name:
                        breakdown['LM Head'] = p.numel()  # 即使共享，也记录原始大小
                    elif 'encoder' in name:
                        breakdown['Encoder'] = p.numel() if 'Encoder' not in breakdown else breakdown[
                                                                                                'Encoder'] + p.numel()
                    elif 'task_embed' in name:
                        breakdown['Task Embed'] = p.numel()
                    elif 'pos_embed' in name:
                        breakdown['Pos Embed'] = p.numel()
                    else:
                        # 其他参数
                        key = name.split('.')[0]
                        breakdown[key] = breakdown.get(key, 0) + p.numel()

        if verbose:
            print("参数 breakdown:")
            for comp, params in breakdown.items():
                print(f"  {comp}: {params / 1e6:.1f}M")
            print(f"总参数量: {total / 1e6:.1f}M (共享已去重)")

        return total


# -----------------------------
# 测试代码（验证模型是否可运行）
# -----------------------------
if __name__ == "__main__":
    model = TinyTransformer()
    print(f"✅ 模型参数量: {model.num_parameters() / 1e6:.1f}M")

    # 仿真输入
    input_ids = torch.randint(0, 32000, (2, 32))
    attention_mask = torch.ones_like(input_ids)  # 全1，无mask
    task_id = torch.tensor([0, 1])  # 任务0和1混合batch

    outputs = model(input_ids, task_id=task_id, attention_mask=attention_mask)
    print(f"✅ 前向计算通过！Logits形状: {outputs['logits'].shape}")
