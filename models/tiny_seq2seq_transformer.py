# tiny_seq2seq_transformer.py
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer
from torch.quantization import QuantStub, DeQuantStub

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ModelConfig


class TinySeq2SeqTransformer(nn.Module):
    """
    🔹 TinySeq2SeqTransformer：轻量级 Encoder-Decoder 翻译模型
    ✅ 动态加载教师 tokenizer（支持 zh→en / en→zh）
    ✅ share_weights 支持
    ✅ Pre-Norm + 可量化设计
    ✅ 端侧友好 (~40M 参数)
    """

    def __init__(
        self,
        teacher_model_path_zh2en=None,
        teacher_model_path_en2zh=None,
        d_model=None,
        nhead=None,
        num_encoder_layers=None,
        num_decoder_layers=None,
        dim_feedforward=None,
        dropout=None,
        max_seq_len=None,
        share_weights=True,
        vocab_size=None,
    ):
        super().__init__()

        cfg = ModelConfig.CURRENT_CONFIG

        # ===== 模型结构参数 =====
        self.d_model = d_model or cfg.get("d_model", 128)
        self.nhead = nhead or cfg.get("nhead", 4)
        self.num_encoder_layers = num_encoder_layers or cfg.get("num_encoder_layers", cfg.get("num_layers", 2))
        self.num_decoder_layers = num_decoder_layers or cfg.get("num_decoder_layers", cfg.get("num_layers", 2))
        self.dim_feedforward = dim_feedforward or cfg.get("dim_feedforward", self.d_model * 2)
        self.dropout = dropout or cfg.get("dropout", 0.1)
        self.max_seq_len = max_seq_len or ModelConfig.MAX_SEQ_LEN
        self.share_weights = share_weights

        # ===== Tokenizer 动态加载 =====
        self.tokenizer_zh2en, self.tokenizer_en2zh = None, None
        zh_vocab, en_vocab = None, None

        if teacher_model_path_zh2en and os.path.exists(teacher_model_path_zh2en):
            self.tokenizer_zh2en = AutoTokenizer.from_pretrained(teacher_model_path_zh2en, local_files_only=True)
            zh_vocab = len(self.tokenizer_zh2en)
            print(f"✅ 成功加载 zh→en 教师 tokenizer, 词表={zh_vocab}")

        if teacher_model_path_en2zh and os.path.exists(teacher_model_path_en2zh):
            self.tokenizer_en2zh = AutoTokenizer.from_pretrained(teacher_model_path_en2zh, local_files_only=True)
            en_vocab = len(self.tokenizer_en2zh)
            print(f"✅ 成功加载 en→zh 教师 tokenizer, 词表={en_vocab}")

        # 取较大 vocab 以兼容双向（embedding 共享）
        if zh_vocab and en_vocab:
            self.vocab_size = max(zh_vocab, en_vocab)
        elif zh_vocab:
            self.vocab_size = zh_vocab
        elif en_vocab:
            self.vocab_size = en_vocab
        else:
            self.vocab_size = vocab_size or ModelConfig.VOCAB_SIZE
            print(f"⚠️ 教师 tokenizer 未找到，使用默认 vocab_size={self.vocab_size}")

        # ===== 模型层定义 =====
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)

        # LM Head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if self.share_weights:
            self.lm_head.weight = self.embed.weight
            print("✅ 启用权重共享模式 (embed.weight = lm_head.weight)")

        # 量化
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self._init_weights()
        print(f"✅ 初始化完成 (vocab={self.vocab_size}, d_model={self.d_model})")

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
    ):
        if encoder_out is not None and encode_only:
            raise ValueError("❌ 不应同时传入 encode_only 和 encoder_out")

        if src_ids is None and input_ids is not None:
            src_ids = input_ids
        if tgt_ids is None and decoder_input_ids is not None:
            tgt_ids = decoder_input_ids

        if src_padding_mask is None and attention_mask is not None:
            src_padding_mask = (attention_mask == 0)
        if tgt_padding_mask is None and decoder_attention_mask is not None:
            tgt_padding_mask = (decoder_attention_mask == 0)

        # ===== Encoder =====
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
        if encoder_out is None:
            encoder_out = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        encoder_out = self.dequant(encoder_out)

        if encode_only:
            return {"encoder_out": encoder_out}

        # ===== Decoder =====
        tgt_emb = self.embed(tgt_ids) * (self.d_model ** 0.5)
        tgt_emb = tgt_emb + self.pos_embed[:, :tgt_ids.size(1), :]
        if task_id is not None:
            tgt_emb = tgt_emb + task_emb
        tgt_emb = self.quant(tgt_emb)

        out = self.decoder(
            tgt_emb,
            encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        out = self.dequant(out)
        logits = self.lm_head(out)
        return {"logits": logits, "encoder_out": encoder_out}

    # --------------------------------------
    @torch.inference_mode()
    def generate(
            self,
            input_ids,
            task_id=0,
            max_length=64,
            num_beams=1,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
            **kwargs,
    ):
        """
        通用生成接口，兼容 Hugging Face 调用格式。

        Args:
            task_id: 0 表示 zh→en，1 表示 en→zh
        """
        device = input_ids.device

        # ✅ 选择 tokenizer
        tok = getattr(self, "tokenizer_zh2en", None) if task_id == 0 else getattr(self, "tokenizer_en2zh", None)
        if tok is None:
            raise ValueError("Tokenizer 未初始化，请确保 TinySeq2SeqTransformer 初始化时传入教师路径。")

        # 修改赋值：先取用户传入，其次 tokenizer 值（如果 None 则跳过），最后默认值
        # 对于 bos：优先 bos_token_id > tok.bos_token_id > tok.cls_token_id > 1
        bos_candidates = [
            bos_token_id,
            getattr(tok, "bos_token_id", None),
            getattr(tok, "cls_token_id", None)
        ]
        bos = next((cand for cand in bos_candidates if cand is not None), 1)  # 如果全 None，用 1

        # 对于 eos：优先 eos_token_id > tok.eos_token_id > tok.sep_token_id > 2
        eos_candidates = [
            eos_token_id,
            getattr(tok, "eos_token_id", None),
            getattr(tok, "sep_token_id", None)
        ]
        eos = next((cand for cand in eos_candidates if cand is not None), 2)  # 如果全 None，用 2

        # 对于 pad 同理，但用 eos 作为最终 fallback
        pad_candidates = [
            pad_token_id,
            getattr(tok, "pad_token_id", None)
        ]
        pad = next((cand for cand in pad_candidates if cand is not None), eos)

        # 额外检查：确保是整数
        if not isinstance(bos, int) or not isinstance(eos, int):
            raise ValueError("BOS 或 EOS token ID 必须是整数，请检查 tokenizer 配置。")

        # ✅ Beam Search 分支
        if num_beams > 1 and hasattr(self, "_generate_beam_search"):
            return self._generate_beam_search(
                input_ids=input_ids,
                task_id=task_id,
                max_len=max_length,
                beam_size=num_beams,
                bos_token_id=bos,
                eos_token_id=eos,
            )

        # ✅ Greedy Search 分支
        generated = torch.full(
            (input_ids.size(0), 1),
            bos,
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_length):
            out = self.forward(src_ids=input_ids, tgt_ids=generated, task_id=task_id)
            next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos).all():
                break

        return generated

    # --------------------------------------
    @torch.no_grad()
    def _generate_beam_search(self, input_ids, task_id, max_len, beam_size, bos_token_id, eos_token_id):
        device = input_ids.device
        B = input_ids.size(0)

        encoder_out = self.forward(src_ids=input_ids, encode_only=True, task_id=task_id)["encoder_out"]

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

            # ---- ✅ 拼接序列 ----
            new_generated = []
            for i in range(B):
                beams = []
                for b in range(beam_size):
                    parent = beam_indices[i, b].item()
                    token = token_indices[i, b].unsqueeze(0).unsqueeze(0)  # [1,1]
                    prev_seq = generated[i * beam_size + parent].unsqueeze(0)  # [1, seq_len]
                    beams.append(torch.cat([prev_seq, token], dim=1))  # ✅ 拼接在 seq_len 维度
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
