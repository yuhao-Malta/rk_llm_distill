# scripts/evaluate_models.py
import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from models.tiny_transformer import TinyTransformer  # 👈 你的学生模型
from sacrebleu import corpus_bleu
import jieba
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# 设置 NLTK 数据目录为项目内的 nltk_data 文件夹
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)  # 👈 添加自定义路径

# 验证 punkt 是否能加载
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt 数据加载成功！")
except LookupError:
    print("❌ NLTK punkt 数据未找到，请检查路径：", NLTK_DATA_DIR)
    raise


# -----------------------------
# 2. 模型加载器
# -----------------------------
class ModelEvaluator:
    def __init__(self, model_name_or_path, model_type="opus_mt"):
        # ✅ 获取项目根目录（绝对路径）
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # ✅ 构建模型绝对路径
        if model_type == "opus_mt":
            model_name_or_path = os.path.join(PROJECT_ROOT, "models", "opus_mt_zh_en")

        self.model_name = model_name_or_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.load_model()

        if self.model_type == "tiny_transformer":
            self.model = TinyTransformer()
            self.model.load_state_dict(torch.load(self.model_name, map_location="cpu"))
            self.model.eval()
            self.tokenizer = self.model.tokenizer  # 复用模型的 tokenizer

    def load_model(self):
        """加载模型和分词器（支持本地加载）"""
        print(f"📥 加载模型: {self.model_name} ({self.model_type})")

        if self.model_type == "opus_mt":
            # ✅ 本地加载 OPUS-MT 模型
            LOCAL_MODEL_PATH = "models/opus_mt_zh_en"  # 👈 你的本地模型路径
            try:
                self.tokenizer = MarianTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True  # 👈 强制离线加载
                )
                self.model = MarianMTModel.from_pretrained(
                    self.model_name,
                    local_files_only=True  # 👈 强制离线加载
                )
                print("✅ OPUS-MT 模型加载成功！")
            except Exception as e:
                print(f"❌ OPUS-MT 模型加载失败: {str(e)}")
                raise
        elif self.model_type == "tiny_transformer":
            # 假设 TinyTransformer 有一个 translate 方法
            self.model = TinyTransformer()
            self.model.load_state_dict(torch.load(self.model_name, map_location="cpu"))
            self.model.eval()
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def translate(self, texts, src_lang="zh", tgt_lang="en"):
        """批量翻译文本"""
        if self.model_type == "opus_mt":
            return self._translate_opus_mt(texts)
        elif self.model_type == "tiny_transformer":
            return self._translate_tiny_transformer(texts)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _translate_opus_mt(self, texts):
        """使用 OPUS-MT 翻译"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _translate_tiny_transformer(self, texts):
        """使用 TinyTransformer 翻译（模拟）"""
        # TODO: 实现 TinyTransformer 的实际翻译逻辑
        # 这里仅为示意，实际应调用模型前向传播
        # print("⚠️  TinyTransformer 翻译逻辑待实现，返回模拟结果...")
        # return [f"[TinyTransformer 翻译]: {text}" for text in texts]
        try:
            # 1. 编码输入文本
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask

            # 2. 判断任务类型（中→英 or 英→中）
            # 简单判断：如果第一个字符是中文，则为中→英
            if '\u4e00' <= texts[0] <= '\u9fff':
                task_id = torch.tensor([0])  # 0 = 中→英
            else:
                task_id = torch.tensor([1])  # 1 = 英→中

            # 3. 模型前向传播
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id
                )
                logits = outputs["logits"]

            # 4. 解码输出（贪心解码）
            predicted_ids = torch.argmax(logits, dim=-1)
            translated = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return translated.strip()

        except Exception as e:
            print(f"❌ TinyTransformer 翻译出错: {str(e)}")
            return "[翻译失败]"


# -----------------------------
# 3. 评估器
# -----------------------------
class TranslationEvaluator:
    def __init__(self, test_data_path="data/test/zh_en.jsonl"):
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()

    def _load_test_data(self):
        """加载测试数据"""
        data = []
        if os.path.exists(self.test_data_path):
            with open(self.test_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            # 如果没有测试数据，生成一些示例
            print("⚠️  未找到测试数据，生成示例数据...")
            data = [
                {"src": "今天天气真好", "ref": "The weather is nice today."},
                {"src": "我喜欢学习人工智能", "ref": "I like to study artificial intelligence."},
                {"src": "瑞芯微正在定义端侧大模型的未来",
                 "ref": "Rockchip is defining the future of on-device large models."},
                {"src": "The weather is nice today.", "ref": "今天天气真好"},
                {"src": "I love artificial intelligence.", "ref": "我喜欢人工智能"},
            ]
            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
            with open(self.test_data_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return data

    def evaluate_model(self, model_evaluator: ModelEvaluator):
        """评估单个模型"""
        print(f"\n🔍 评估模型: {model_evaluator.model_name}")
        translations = []
        references = []

        for item in self.test_data:
            src_text = item["src"]
            ref_text = item["ref"]

            # 自动检测语言（简化版）
            if len(src_text) > 0 and '\u4e00' <= src_text[0] <= '\u9fff':
                src_lang, tgt_lang = "zh", "en"
            else:
                src_lang, tgt_lang = "en", "zh"

            hyp_text = model_evaluator.translate([src_text], src_lang, tgt_lang)[0]
            translations.append(hyp_text)
            references.append(ref_text)
            print(f"原文: {src_text}")
            print(f"参考: {ref_text}")
            print(f"翻译: {hyp_text}\n")

        # 计算 BLEU
        bleu_score = corpus_bleu(translations, [references]).score
        print(f"✅ BLEU 分数: {bleu_score:.2f}")
        return {
            "model_name": model_evaluator.model_name,
            "bleu": bleu_score,
            "translations": translations,
            "references": references
        }

    def compare_models(self, evaluators: list):
        """对比多个模型"""
        results = []
        for evaluator in evaluators:
            result = self.evaluate_model(evaluator)
            results.append(result)

        # 打印对比表格
        print("\n📊 模型对比结果:")
        print("-" * 80)
        print(f"{'模型名称':<30} {'BLEU 分数':<10}")
        print("-" * 80)
        for result in results:
            print(f"{result['model_name']:<30} {result['bleu']:<10.2f}")
        print("-" * 80)


# -----------------------------
# 4. 主程序入口
# -----------------------------
if __name__ == "__main__":
    # 1. 初始化评估器
    evaluator = TranslationEvaluator()

    # 2. 定义要评估的模型列表
    model_evaluators = [
        ModelEvaluator("../models/opus_mt_zh_en", model_type="opus_mt"),  # 👈 改为本地路径
        ModelEvaluator("../outputs/models/student_model_amp_epoch_3.pth", model_type="tiny_transformer"),  # 👈 替换为你的模型路径
    ]

    # 3. 开始对比评估
    evaluator.compare_models(model_evaluators)