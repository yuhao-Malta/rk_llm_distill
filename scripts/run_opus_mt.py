import os
import torch
from transformers import MarianMTModel, MarianTokenizer


class OPUSMTTranslator:
    def __init__(self):
        # 加载中→英模型
        self.model_zh_en = MarianMTModel.from_pretrained("models/opus_mt_zh_en")
        self.tokenizer_zh_en = MarianTokenizer.from_pretrained("models/opus_mt_zh_en")

        # 加载英→中模型 👈 新增！
        self.model_en_zh = MarianMTModel.from_pretrained("models/opus_mt_en_zh")
        self.tokenizer_en_zh = MarianTokenizer.from_pretrained("models/opus_mt_en_zh")

    def translate_zh_to_en(self, texts):
        """中文→英文"""
        encoded = self.tokenizer_zh_en(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            generated = self.model_zh_en.generate(**encoded)
        return self.tokenizer_zh_en.batch_decode(generated, skip_special_tokens=True)

    def translate_en_to_zh(self, texts):
        """英文→中文 👈 新增！"""
        encoded = self.tokenizer_en_zh(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            generated = self.model_en_zh.generate(**encoded)
        return self.tokenizer_en_zh.batch_decode(generated, skip_special_tokens=True)


# 测试翻译
translator = OPUSMTTranslator()

# 中→英
zh_texts = [
    "今天天气真好",
    "我喜欢学习人工智能",
    "瑞芯微正在定义端侧大模型的未来"
]

print("🇨🇳→🇺🇸 中文→英文:")
for src, tgt in zip(zh_texts, translator.translate_zh_to_en(zh_texts)):
    print(f"原文: {src}")
    print(f"翻译: {tgt}\n")

# 英→中
en_texts = [
    "In which of these following situations should you avoid overtaking?",
    "What do child locks in a vehicle do?",
    "Stop children from opening rear doors from the inside"
]

print("🇺🇸→🇨🇳 英文→中文:")
for src, tgt in zip(en_texts, translator.translate_en_to_zh(en_texts)):
    print(f"原文: {src}")
    print(f"翻译: {tgt}\n")