import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer
from torch.quantization import QuantStub, DeQuantStub

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ModelConfig, MODEL_PATH

class TinySeq2SeqTransformer(nn.Module):
    """
    🔹 TinySeq2SeqTransformer：轻量级 Encoder-Decoder 翻译模型
    特点：
      ✅ 双向任务 (zh→en / en→zh)
      ✅ share_weights 支持
      ✅ Pre-Norm + 可量化设计
      ✅ 端侧友好（~40M 参数，seq_len=64）
    """

    def __init__(
            self,
            d_model=None,
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            dropout=None,
            max_seq_len=None,
            share_weights=None,
            vocab_size=None
    ):
        super().__init__()

        cfg = ModelConfig.CURRENT_CONFIG

        # ========== 参数解析 ==========
        self.d_model = d_model or cfg.get("d_model", 128)
        self.nhead = nhead or cfg.get("nhead", 4)
        self.num_encoder_layers = num_encoder_layers or cfg.get("num_encoder_layers", cfg.get("num_layers", 2))
        self.num_decoder_layers = num_decoder_layers or cfg.get("num_decoder_layers", cfg.get("num_layers", 2))
        self.dim_feedforward = dim_feedforward or cfg.get("dim_feedforward", self.d_model * 2)
        self.dropout = dropout or cfg.get("dropout", 0.1)
        self.max_seq_len = max_seq_len or ModelConfig.MAX_SEQ_LEN
        self.share_weights = share_weights if share_weights is not None else cfg.get("share_weights", True)
        self.vocab_size = vocab_size or ModelConfig.VOCAB_SIZE

        # ========== Tokenizer ==========
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Tokenizer路径不存在: {MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        print(f"✅ 成功加载 Qwen Tokenizer，词表大小={len(self.tokenizer)}")

        actual_vocab = len(self.tokenizer)
        if actual_vocab != self.vocab_size:
            print(f"⚠️ 词汇表大小不匹配: config={self.vocab_size}, actual={actual_vocab}")
            self.vocab_size = actual_vocab
            ModelConfig.VOCAB_SIZE = actual_vocab

        # ========== 模型层 ==========
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        self.task_embed = nn.Embedding(2, self.d_model)  # 0=zh→en, 1=en→zh

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)

        # LM Head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if self.share_weights:
            self.lm_head.weight = self.embed.weight
            print("✅ 启用权重共享模式 (embed.weight = lm_head.weight)")

        # 量化支持
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self._init_weights()

    # --------------------------------------
    def _init_weights(self):
        """权重初始化（适配共享权重和量化）"""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.task_embed.weight, mean=0.0, std=0.02)

        # 如果未共享权重，则单独初始化输出层
        if not self.share_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        print(f"✅ 权重初始化完成 (d_model={self.d_model}, std=0.02)")

    def forward(
            self,
            src_ids=None,
            tgt_ids=None,
            task_id=None,
            encode_only=False,
            encoder_out=None,
            src_mask=None,
            tgt_mask=None,
            src_padding_mask=None,
            tgt_padding_mask=None,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            **kwargs
    ):
        """
        通用 forward：
          - 支持 encode_only=True（仅编码器输出）
          - 兼容蒸馏训练与推理
        """
        if encoder_out is not None and encode_only:
            raise ValueError("❌ 不应该同时传入 encode_only 和 encoder_out！")

        if encoder_out is not None:
            print("[DEBUG] ✅ 使用缓存的 encoder_out，跳过 encoder")
        else:
            print("[DEBUG] ⚠️ 重新计算 encoder（未缓存）")

        if encode_only:
            src_emb = self.embed(src_ids) * (self.d_model ** 0.5)
            src_emb = src_emb + self.pos_embed[:, :src_ids.size(1), :]
            src_emb = self.quant(src_emb)
            memory = self.encoder(src_emb)
            memory = self.dequant(memory)
            return {"encoder_out": memory}

        # 自动适配输入
        if src_ids is None and input_ids is not None:
            src_ids = input_ids
        if tgt_ids is None and decoder_input_ids is not None:
            tgt_ids = decoder_input_ids

        if src_padding_mask is None and attention_mask is not None:
            src_padding_mask = (attention_mask == 0)
        if tgt_padding_mask is None and decoder_attention_mask is not None:
            tgt_padding_mask = (decoder_attention_mask == 0)

        # ==== Encoder ====
        src_emb = self.embed(src_ids) * (self.d_model ** 0.5)
        src_emb = src_emb + self.pos_embed[:, :src_ids.size(1), :]

        if task_id is not None:
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.long, device=src_ids.device)
            if task_id.dim() == 0:
                task_id = task_id.unsqueeze(0).expand(src_ids.size(0))
            task_emb = self.task_embed(task_id).unsqueeze(1)
            src_emb = src_emb + task_emb

        src_emb = self.quant(src_emb)
        if encoder_out is not None:
            memory = encoder_out  # ✅ 直接使用缓存好的 encoder 输出
        else:
            memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # ==== Decoder ====
        if tgt_ids is not None:
            tgt_emb = self.embed(tgt_ids) * (self.d_model ** 0.5)
            tgt_emb = tgt_emb + self.pos_embed[:, :tgt_ids.size(1), :]
            if task_id is not None:
                tgt_emb = tgt_emb + task_emb
            tgt_emb = self.quant(tgt_emb)
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
        else:
            out = memory

        out = self.dequant(out)
        logits = self.lm_head(out)
        return {"logits": logits, "encoder_out": memory}

    # --------------------------------------
    def  _forward_internal(
            self,
            src_ids,
            tgt_ids=None,
            task_id=None,
            src_mask=None,
            tgt_mask=None,
            src_padding_mask=None,
            tgt_padding_mask=None
    ):
        """
        Seq2Seq 前向传播（支持蒸馏训练与推理）

        :param src_ids: [batch, src_len]
        :param tgt_ids: [batch, tgt_len]（蒸馏时 teacher 生成）
        :param task_id: int or Tensor (0=zh→en, 1=en→zh)
        """
        B, S = src_ids.size()
        device = src_ids.device

        # ===== Embedding =====
        src_emb = self.embed(src_ids) * (self.d_model ** 0.5)
        src_emb = src_emb + self.pos_embed[:, :S, :]

        # 任务嵌入
        if task_id is not None:
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor(task_id, dtype=torch.long, device=device)
            if task_id.dim() == 0:
                task_id = task_id.unsqueeze(0).expand(B)
            task_emb = self.task_embed(task_id).unsqueeze(1)  # [B,1,d_model]
            src_emb = src_emb + task_emb

        # ===== Encoder =====
        src_emb = self.quant(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # ===== Decoder =====
        if tgt_ids is not None:
            _, T = tgt_ids.size()
            tgt_emb = self.embed(tgt_ids) * (self.d_model ** 0.5)
            tgt_emb = tgt_emb + self.pos_embed[:, :T, :]
            if task_id is not None:
                tgt_emb = tgt_emb + task_emb

            tgt_emb = self.quant(tgt_emb)
            out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            out = self.dequant(out)
            logits = self.lm_head(out)
            logits = logits.contiguous()  # 🔧 确保量化后连续
            return {"logits": logits}

        else:
            # 蒸馏/编码器输出模式
            memory = self.dequant(memory)
            return {"encoder_output": memory}

    @torch.inference_mode()
    def generate(
            self,
            input_ids=None,
            task_id=0,
            max_length=64,
            num_beams=1,
            bos_token_id=None,
            eos_token_id=None,
            **kwargs,
    ):
        device = input_ids.device

        # tokenizer 自动取特殊符号
        if hasattr(self, "tokenizer"):
            tok = self.tokenizer
            bos_token_id = bos_token_id or tok.bos_token_id or tok.cls_token_id
            eos_token_id = eos_token_id or tok.eos_token_id or tok.sep_token_id
        else:
            bos_token_id = bos_token_id or 151643
            eos_token_id = eos_token_id or 151643

        if num_beams > 1:
            return self._generate_beam_search(
                input_ids=input_ids,
                task_id=task_id,
                max_len=max_length,
                beam_size=num_beams,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )

        # ---- Greedy ----
        B = input_ids.size(0)
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        for _ in range(max_length):
            out = self.forward(src_ids=input_ids, tgt_ids=generated, task_id=task_id)
            next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return generated

    # --------------------------------------
    def num_parameters(self, verbose=False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f"📊 模型总参数: {total/1e6:.2f}M")
        return total

    @torch.no_grad()
    def _generate_beam_search(
            self,
            input_ids,
            task_id=0,
            max_len=64,
            beam_size=4,
            bos_token_id=None,
            eos_token_id=None,
    ):
        """
        高效 Beam Search（支持 encoder 缓存 + 正确终止逻辑）
        """
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        bos_token_id = getattr(self, "bos_token_id", None)
        eos_token_id = getattr(self, "eos_token_id", None)
        pad_token_id = getattr(self, "pad_token_id", eos_token_id)

        if bos_token_id is None or eos_token_id is None:
            raise ValueError("❌ 模型未设置 bos_token_id/eos_token_id，请在加载后手动注入。")

        # ---- 1. 编码器 ----
        encoder_out = self.forward(src_ids=input_ids, encode_only=True)["encoder_out"]

        generated = torch.full((B * beam_size, 1), bos_token_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B, beam_size, device=device)
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            out = self.forward(
                src_ids=input_ids.repeat_interleave(beam_size, dim=0),
                tgt_ids=generated,
                task_id=task_id,
                encoder_out=encoder_out.repeat_interleave(beam_size, dim=0),
            )
            logits = out["logits"][:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = log_probs.size(-1)

            scores = beam_scores.view(B, beam_size, 1) + log_probs.view(B, beam_size, vocab_size)
            top_scores, top_pos = torch.topk(scores.view(B, -1), beam_size, dim=-1)
            beam_indices = top_pos // vocab_size
            token_indices = top_pos % vocab_size

            beam_scores = top_scores

            # 拼接新序列
            new_generated = []
            for i in range(B):
                beams = []
                for b in range(beam_size):
                    parent = beam_indices[i, b].item()
                    token = token_indices[i, b].view(1, 1)
                    prev_seq = generated[i * beam_size + parent]
                    beams.append(torch.cat([prev_seq, token], dim=0).unsqueeze(0))
                new_generated.extend(beams)
            generated = torch.cat(new_generated, dim=0)

            # EOS 检查
            eos_mask = (generated[:, -1] == eos_token_id)
            finished = finished | eos_mask
            if finished.all():
                break

        # ---- 3. 选出最优 beam ----
        generated = generated.view(B, beam_size, -1)
        beam_scores = beam_scores.view(B, beam_size)
        best_idx = torch.argmax(beam_scores, dim=1)
        best_seq = generated[torch.arange(B, device=device), best_idx]
        return best_seq

# --------------------------------------
# ✅ 测试
if __name__ == "__main__":
    print("\n🧪 测试 TinySeq2SeqTransformer\n")
    model = TinySeq2SeqTransformer(share_weights=True)

    batch, src_len, tgt_len = 2, 16, 16
    src = torch.randint(0, ModelConfig.VOCAB_SIZE, (batch, src_len))
    tgt = torch.randint(0, ModelConfig.VOCAB_SIZE, (batch, tgt_len))

    with torch.no_grad():
        out = model(src, tgt, task_id=torch.tensor([0, 1]))
        print("✅ 前向传播成功")
        print(f"  logits: {out['logits'].shape}")  # [B, T, V]
