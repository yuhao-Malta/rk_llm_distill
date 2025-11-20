#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_student_alignment.py
用于检查学生模型与教师模型在 tokenizer、embedding、输出上的对齐情况。
"""
import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from models.tiny_seq2seq_transformer import TinySeq2SeqTransformer
from config.config import ModelConfig, EvalConfig, MODEL_PATH, OPUS_MT_ZH_EN, OPUS_MT_EN_ZH

# ==== 1️⃣ 路径配置 ====
TEACHER_PATH = OPUS_MT_ZH_EN  # ← 改成 teacher 模型目录
STUDENT_MODEL_PATH = "outputs/models/student_model_final_merged.pth"
STUDENT_TOKENIZER_PATH = OPUS_MT_ZH_EN  # 或 "outputs/tokenizer"

# ==== 2️⃣ 加载 Tokenizer ====
print("\n🔍 Loading tokenizers ...")
teacher_tok = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
student_tok = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER_PATH, trust_remote_code=True)

print(f"Teacher vocab size: {len(teacher_tok)}")
print(f"Student vocab size: {len(student_tok)}")

if len(teacher_tok) != len(student_tok):
    print(f"⚠️ 词表大小不匹配: teacher={len(teacher_tok)}, student={len(student_tok)}")

print("\n🧩 前20个token对齐检查：")
for i in range(20):
    t_tok = teacher_tok.convert_ids_to_tokens(i)
    s_tok = student_tok.convert_ids_to_tokens(i)
    marker = "✅" if t_tok == s_tok else "❌"
    print(f"{i:>5}: {t_tok:<15} | {s_tok:<15} {marker}")

# # 如果 tokenizer 没有 bos/eos，就指定合理的默认值
# if student_tok.bos_token_id is None:
#     student_tok.bos_token = "<s>"
#     student_tok.bos_token_id = 151642  # 你可以查看 tokenizer.vocab_size 附近的保留符号
# if student_tok.eos_token_id is None:
#     student_tok.eos_token = "</s>"
#     student_tok.eos_token_id = 151643  # 你上次日志中 pad/eos=151643，很可能就是它
# ==== 3️⃣ 加载学生模型 ====
print("\n🧠 Loading student model ...")
device = "cpu"
model = TinySeq2SeqTransformer(
                teacher_model_path_zh2en=OPUS_MT_ZH_EN,
                teacher_model_path_en2zh=OPUS_MT_EN_ZH,
                d_model=ModelConfig.CURRENT_CONFIG.get("d_model", 128),
                nhead=ModelConfig.CURRENT_CONFIG.get("nhead", 4),
                num_encoder_layers=ModelConfig.CURRENT_CONFIG.get("num_encoder_layers", 2),
                num_decoder_layers=ModelConfig.CURRENT_CONFIG.get("num_decoder_layers", 2),
                dim_feedforward=ModelConfig.CURRENT_CONFIG.get("dim_feedforward", 256),
                dropout=ModelConfig.CURRENT_CONFIG.get("dropout", 0.1),
                max_seq_len=ModelConfig.MAX_SEQ_LEN,
                share_weights=True,
            ).to(device)
model.bos_token_id = student_tok.bos_token_id
model.eos_token_id = student_tok.eos_token_id
model.tokenizer = student_tok  # 给 generate 用
state = torch.load(STUDENT_MODEL_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

embed_weight = model.embed.weight
print(f"📐 Student embedding shape: {embed_weight.shape}")

if embed_weight.shape[0] != len(student_tok):
    print(f"⚠️ 嵌入层与 tokenizer 不匹配: embed={embed_weight.shape[0]}, vocab={len(student_tok)}")
else:
    print("✅ 嵌入层与 tokenizer 大小匹配")

# ==== 4️⃣ 测试翻译 ====
print("\n🧪 翻译对照测试 ...")

model.tokenizer = student_tok
model.bos_token_id = getattr(student_tok, "bos_token_id", 151642)
model.eos_token_id = getattr(student_tok, "eos_token_id", 151643)
model.eval()

text = "你好，世界！"
enc = student_tok(text, return_tensors="pt", padding=False)
with torch.no_grad():
    out = model.generate(input_ids=enc["input_ids"], num_beams=4, max_length=64)
decoded = student_tok.decode(out[0], skip_special_tokens=True)
print(f"👩‍🎓 学生模型输出: {decoded}")

# ==== 5️⃣ Teacher 输出对比 ====
try:
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    teacher_enc = teacher_tok(text, return_tensors="pt")
    with torch.no_grad():
        t_out = teacher_model.generate(**teacher_enc, max_length=64)
    t_decoded = teacher_tok.decode(t_out[0], skip_special_tokens=True)
    print(f"👩‍🏫 教师模型输出: {t_decoded}")
except Exception as e:
    print("⚠️ 无法加载教师模型:", e)

print("\n✅ 检查完成！根据上面输出判断：")
print(" - 如果 tokenizer 前20个对不上 → token错位；")
print(" - 如果 vocab 大小不同 → tokenizer不匹配；")
print(" - 如果 teacher 输出正常、student 全乱码 → 学生没学到。")