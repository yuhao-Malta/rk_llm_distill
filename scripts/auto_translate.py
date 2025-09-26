import os
import torch
import fasttext  # 👈 新增导入
from transformers import MarianMTModel, MarianTokenizer


class AutoTranslator:
    def __init__(self):
        """初始化中英双向翻译器"""
        print("🚀 正在加载 OPUS-MT 中英双向模型...")

        # 获取项目根目录
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 加载中→英模型
        zh_en_path = os.path.join(PROJECT_ROOT, "models", "opus_mt_zh_en")
        self.model_zh_en = MarianMTModel.from_pretrained(zh_en_path, local_files_only=True)
        self.tokenizer_zh_en = MarianTokenizer.from_pretrained(zh_en_path, local_files_only=True)

        # 加载英→中模型
        en_zh_path = os.path.join(PROJECT_ROOT, "models", "opus_mt_en_zh")
        self.model_en_zh = MarianMTModel.from_pretrained(en_zh_path, local_files_only=True)
        self.tokenizer_en_zh = MarianTokenizer.from_pretrained(en_zh_path, local_files_only=True)

        # 👇 加载 fasttext 语言检测模型
        model_path = os.path.join(PROJECT_ROOT, "models", "lid.176.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ fasttext 语言检测模型不存在！请下载 lid.176.bin 到 models/ 目录")

        self.lang_detector = fasttext.load_model(model_path)
        print("✅ 模型加载完成！")

    def detect_language(self, text: str) -> str:
        """
        使用 fasttext 检测文本语言
        :param text: 输入文本
        :return: 'zh' (中文) 或 'en' (英文)
        """
        try:
            # 预测语言（返回概率最高的语言）
            predictions = self.lang_detector.predict(text, k=1)  # k=1 表示只返回最可能的语言
            lang_code = predictions[0][0].replace('__label__', '')  # 去掉 __label__ 前缀

            # 映射到 zh/en
            if lang_code in ['zh', 'zh-cn', 'zh-tw']:
                return 'zh'
            elif lang_code == 'en':
                return 'en'
            else:
                print(f"⚠️  未知语言 '{lang_code}'，默认按中文处理")
                return 'zh'
        except Exception as e:
            print(f"❌ 语言检测失败: {str(e)}，默认按中文处理")
            return 'zh'

    def translate(self, text: str) -> str:
        """自动检测语言并翻译"""
        src_lang = self.detect_language(text)

        if src_lang == 'zh':
            print("🇨🇳 检测到中文，翻译成英文...")
            model = self.model_zh_en
            tokenizer = self.tokenizer_zh_en
        else:
            print("🇺🇸 检测到英文，翻译成中文...")
            model = self.model_en_zh
            tokenizer = self.tokenizer_en_zh

        try:
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                generated = model.generate(**encoded)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return translated
        except Exception as e:
            print(f"❌ 翻译出错: {str(e)}")
            return "翻译失败"


if __name__ == "__main__":
    translator = AutoTranslator()

    test_texts = [
        "今天天气真好",
        "I love artificial intelligence",
        "瑞芯微正在定义端侧大模型的未来",
        "What is the capital of France?"
    ]

    print("🔍 自动语言检测 + 翻译测试:")
    print("-" * 50)

    for text in test_texts:
        print(f"\n原文: {text}")
        result = translator.translate(text)
        print(f"翻译: {result}")