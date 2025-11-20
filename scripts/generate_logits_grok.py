import os
import sys
import torch
import psutil
import json
import gc

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import logging
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ModelConfig, DataFormat, LogConfig,
    OPUS_MT_ZH_EN, OPUS_MT_EN_ZH, TEACHER_LOGITS_DIR, RAW_DATA_PATH
)

try:
    import dashscope
except ImportError:
    dashscope = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format=LogConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LogConfig.GENERATE_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# DashScope API Key
DASHSCOPE_API_KEY = "sk-b0c78a77e5ea489b8c68e0b5049204c6"  # 请替换


# 检查模型文件
def check_model_files(model_path):
    required_files = ["config.json", "tokenizer_config.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            raise FileNotFoundError(f"❌ 缺少模型文件: {os.path.join(model_path, file)}")
    # 检查模型权重
    if not (any(f.endswith("pytorch_model.bin") for f in os.listdir(model_path)) or
            any(f.endswith(".safetensors") for f in os.listdir(model_path))):
        raise FileNotFoundError(f"❌ 模型权重文件（pytorch_model.bin或safetensors）不存在: {model_path}")
    # 检查safetensors大小
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if safetensors_files:
        size_mb = os.path.getsize(os.path.join(model_path, safetensors_files[0])) / 1024 ** 2
        logging.info(f"✅ 模型权重文件: {safetensors_files[0]}, 大小: {size_mb:.2f} MB")
    logging.info(f"✅ 模型文件检查通过: {model_path}")


# DashScope API翻译
# def call_qwen_translate_api(text, target_lang="en", max_retries=3):
#     if not dashscope:
#         raise ImportError("❌ DashScope未安装，请运行 'pip install dashscope'")
#     dashscope.api_key = DASHSCOPE_API_KEY
#     TEXT_TRANSLATION_AVAILABLE = hasattr(dashscope, 'TextTranslation')
#
#     for attempt in range(max_retries):
#         logging.info(f"API 翻译尝试 {attempt + 1}/{max_retries}")
#         try:
#             if TEXT_TRANSLATION_AVAILABLE:
#                 response = dashscope.TextTranslation.call(
#                     model='qwen-max',
#                     text=text,
#                     target_language=target_lang
#                 )
#                 if response.status_code == 200:
#                     return response.output['translated_text']
#                 logging.warning(f"⚠️ TextTranslation失败: {response.message}")
#             else:
#                 source_lang_full = "中文" if target_lang == "en" else "英文"
#                 target_lang_full = "英文" if target_lang == "en" else "中文"
#                 prompt = (
#                     f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
#                     f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
#                     f"只输出翻译结果：\n\n{text}"
#                 )
#                 response = dashscope.Generation.call(
#                     model='qwen-max',
#                     prompt=prompt,
#                     max_tokens=512,
#                     temperature=0.1
#                 )
#                 if response.status_code == 200:
#                     return response.output.get('text', '').strip()
#                 logging.warning(f"⚠️ Generation失败: {response.message}")
#         except Exception as e:
#             logging.warning(f"⚠️ API调用异常: {e}")
#         if attempt < max_retries - 1:
#             time.sleep(2 ** attempt)
#     logging.error(f"❌ API翻译失败（{max_retries}次尝试）")
#     return None


# 本地模型翻译
# def call_qwen_translate_local(model, tokenizer, text, target_lang="en", max_seq_len=64):
#     source_lang_full = "中文" if target_lang == "en" else "英文"
#     target_lang_full = "英文" if target_lang == "en" else "中文"
#     prompt = (
#         f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
#         f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
#         f"只输出翻译结果：\n\n{text}"
#     )
#     inputs = tokenizer(
#         prompt, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
#     ).to(model.device)
#     inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
#     inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.logits  # 返回logits


# 自定义数据集类
class TranslationDataset(Dataset):
    """
    翻译数据集（统一格式）
    输出字段严格遵循 DataFormat.REQUIRED_KEYS
    """

    def __init__(self, dataset, tokenizer, max_seq_len=64, src_lang="zh", tgt_lang="en", task_id=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        # 构造翻译提示词
        source_lang_full = "中文" if self.src_lang == "zh" else "英文"
        target_lang_full = "英文" if self.tgt_lang == "en" else "中文"
        prompt = (
            f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
            f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
            f"只输出翻译结果：\n\n{src_text}"
        )

        # 分词
        src_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # ✅ 统一输出格式 (遵循 DataFormat)
        return {
            "id": idx,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_input_ids": src_encoding["input_ids"].squeeze(0).to(dtype=torch.long),
            "src_attention_mask": src_encoding["attention_mask"].squeeze(0).to(dtype=torch.long),
            "tgt_input_ids": tgt_encoding["input_ids"].squeeze(0).to(dtype=torch.long),
            "tgt_attention_mask": tgt_encoding["attention_mask"].squeeze(0).to(dtype=torch.long),
            "task_id": self.task_id
        }


# ==================== 自定义 collate_fn ====================
def custom_collate_fn(batch, max_seq_len=64, pad_token_id=151643):
    """批次数据整理 (统一格式)"""
    keys = ["id", "src_input_ids", "src_attention_mask", "tgt_input_ids", "tgt_attention_mask", "task_id"]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    # Pad 序列
    src_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["src_input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )[:, :max_seq_len].to(dtype=torch.long)

    src_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["src_attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0
    )[:, :max_seq_len].to(dtype=torch.long)

    tgt_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["tgt_input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )[:, :max_seq_len].to(dtype=torch.long)

    tgt_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["tgt_attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0
    )[:, :max_seq_len].to(dtype=torch.long)

    task_ids = torch.tensor([item["task_id"] for item in batch], dtype=torch.long)
    ids = torch.tensor([item["id"] for item in batch], dtype=torch.long)

    return {
        "id": ids,
        "src_text": src_texts,
        "tgt_text": tgt_texts,
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_mask,
        "tgt_input_ids": tgt_input_ids,
        "tgt_attention_mask": tgt_attention_mask,
        "task_id": task_ids
    }


# 验证 logits 文件
def validate_logits_file(file_path, use_api=False, model_path=None):
    """验证生成的.pt或.jsonl文件，API模式检查hyp_text，非API模式动态检查logits维度"""
    try:
        # 动态获取词汇表大小
        if not use_api and model_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            vocab_size = config.get("vocab_size", len(tokenizer))
            logging.info(f"✅ 模型词汇表大小: {vocab_size}")
        else:
            vocab_size = None

        if file_path.endswith(".jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            count = len(data)
            for item in data:
                assert "id" in item and "task_id" in item, f"缺少id或task_id: {item}"
                assert "src" in item and "ref" in item and "hyp" in item, f"缺少src/ref/hyp: {item}"
                assert isinstance(item["src"], str) and isinstance(item["ref"], str) and isinstance(item["hyp"],
                                                                                                    str), f"src/ref/hyp类型错误: {item}"
            logging.info(f"✅ 验证通过！{file_path} 共 {count} 条有效记录（JSONL，API模式）")
        else:
            data = torch.load(file_path)
            count = len(data)
            for item in data:
                assert "id" in item and "task_id" in item, f"缺少id或task_id: {item}"
                if use_api:
                    assert "hyp_text" in item and item["hyp_text"] is not None, f"API模式缺少hyp_text: {item}"
                    logging.warning("⚠️ API模式：logits为None，仅保存翻译文本")
                else:
                    assert "logits" in item and item["logits"] is not None, f"缺少logits: {item}"
                    if vocab_size:
                        if item["logits"].shape[-1] != vocab_size:
                            logging.warning(
                                f"⚠️ logits维度不匹配 (预期: {vocab_size}, 实际: {item['logits'].shape[-1]})")
                        else:
                            logging.debug(f"✅ logits维度验证通过: {item['logits'].shape}")
                    else:
                        logging.debug(f"⚠️ 未提供model_path，跳过logits维度检查，实际维度: {item['logits'].shape}")
            logging.info(f"✅ 验证通过！{file_path} 共 {count} 条有效记录（PT，{'API模式' if use_api else '本地模式'}）")
        return count
    except Exception as e:
        logging.error(f"❌ 验证失败: {e}")
        return 0

# # ==================== 生成 Teacher Logits (主函数) ====================
# def generate_teacher_logits(args):
#     """
#     增强版：生成 Teacher 模型 logits 文件
#     ✅ 加入显存优化 / 自动 batch_size 回退 / 死锁防护 / 半精度支持
#     """
#     process = psutil.Process()
#     logging.info(f"初始内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
#
#     # ===== 1. 加载 tokenizer =====
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(
#             MODEL_PATH, local_files_only=True, trust_remote_code=True
#         )
#         logging.info("✅ 成功加载 Qwen Tokenizer")
#     except Exception as e:
#         logging.error(f"❌ Tokenizer加载失败: {e}")
#         raise
#
#     # ===== 2. 加载模型 =====
#     model = None
#     device = torch.device(args.device)
#     if not args.use_api:
#         try:
#             check_model_files(MODEL_PATH)
#             logging.info(f"📥 加载 Qwen 模型到 {device}...")
#
#             try:
#                 model = AutoModelForCausalLM.from_pretrained(
#                     MODEL_PATH,
#                     device_map=device,
#                     torch_dtype=torch.float32,  # ✅ 改为优先使用 FP32 精度
#                     local_files_only=True,
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=False,  # FP32 模式下禁用低内存加载，避免截断
#                     use_safetensors=True
#                 )
#                 logging.info("✅ 成功加载模型 (float32 全精度)")
#             except Exception as e:
#                 logging.warning(f"⚠️ 加载 float32 模型失败，尝试使用 float16: {e}")
#                 model = AutoModelForCausalLM.from_pretrained(
#                     MODEL_PATH,
#                     device_map=device,
#                     torch_dtype=torch.float16,
#                     local_files_only=True,
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True,
#                     use_safetensors=True
#                 )
#                 logging.info("✅ 回退到 float16 半精度模型")
#
#             model.eval()
#
#             # ✅ 启用 TF32 + cuDNN benchmark
#             if device.type == "cuda":
#                 torch.backends.cuda.matmul.allow_tf32 = True
#                 torch.backends.cudnn.benchmark = True
#                 logging.info("💡 启用 TF32 与 cuDNN Benchmark 以优化 CUDA 稳定性")
#                 logging.info("📊 初始CUDA内存摘要：")
#                 logging.info(torch.cuda.memory_summary(device=device, abbreviated=True))
#         except Exception as e:
#             logging.error(f"❌ 模型加载失败: {e}")
#             raise
#
#     # ===== 3. 加载数据集 =====
#     try:
#         dataset_path = os.path.join(RAW_DATA_PATH, "train/*.parquet")
#         dataset = load_dataset("parquet", data_files={"train": dataset_path})["train"]
#         total_samples = len(dataset)
#         logging.info(f"✅ 加载WMT19数据集，样本数: {total_samples}")
#     except Exception as e:
#         logging.error(f"❌ 数据集加载失败: {e}")
#         raise
#
#     total_samples = min(args.max_samples or total_samples, total_samples)
#     shard_size = min(args.shard_size, total_samples)
#
#     success_count, fail_count = 0, 0
#
#     # ===== 4. 双向翻译任务 =====
#     for src_lang, tgt_lang, task_id, output_prefix in [
#         ("zh", "en", 0, os.path.join(TEACHER_LOGITS_DIR, "zh_to_en")),
#         ("en", "zh", 1, os.path.join(TEACHER_LOGITS_DIR, "en_to_zh"))
#     ]:
#         logging.info(f"🧠 生成 {src_lang}→{tgt_lang} logits (任务ID: {task_id})")
#
#         start_idx = args.start_from
#         end_idx = min(start_idx + args.shard_size, total_samples)
#         shard_dataset = dataset.select(range(start_idx, end_idx))
#         if len(shard_dataset) == 0:
#             logging.warning(f"⚠️ 分片 {args.shard_idx} 无样本，跳过。")
#             continue
#
#         # 构建 DataLoader
#         shard_dataloader = DataLoader(
#             TranslationDataset(
#                 shard_dataset, tokenizer,
#                 max_seq_len=args.max_seq_len,
#                 src_lang=src_lang, tgt_lang=tgt_lang, task_id=task_id
#             ),
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=(device == "cuda"),
#             collate_fn=lambda b: custom_collate_fn(
#                 b, max_seq_len=args.max_seq_len, pad_token_id=tokenizer.pad_token_id
#             )
#         )
#
#         output_file = f"{output_prefix}_shard_{args.shard_idx}.{'jsonl' if args.use_api else 'pt'}"
#         output_data = []
#
#         # == == = 5. 主循环 == == =
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(
#                     tqdm(shard_dataloader, desc=f"{src_lang}→{tgt_lang} 分片 {args.shard_idx}")
#             ):
#                 batch_start = time.time()
#                 try:
#                     src_input_ids = batch["src_input_ids"].to(device)
#                     src_attention_mask = batch["src_attention_mask"].to(device)
#                     tgt_input_ids = batch["tgt_input_ids"].to(device)
#                     tgt_attention_mask = batch["tgt_attention_mask"].to(device)
#
#                     # ============================================================
#                     # ✅ 1️⃣ 区分模型类型
#                     # ============================================================
#                     model_type = getattr(model.config, "model_type", "").lower()
#
#                     if "qwen" in model_type or "llama" in model_type or "mistral" in model_type:
#                         # ============================================================
#                         # ✅ CausalLM 型 (如 Qwen)：拼接 src+tgt，手动shift labels
#                         # ============================================================
#                         input_ids = torch.cat([src_input_ids, tgt_input_ids], dim=1)
#                         attention_mask = torch.cat([src_attention_mask, tgt_attention_mask], dim=1)
#
#                         # 构造 labels，使得模型只预测 target 段
#                         labels = input_ids.clone()
#                         labels[:, :src_input_ids.size(1)] = -100  # 忽略源句部分的loss
#
#                         with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.float32):
#                             outputs = model(
#                                 input_ids=input_ids,
#                                 attention_mask=attention_mask,
#                                 labels=labels,
#                                 output_hidden_states=False,
#                                 output_attentions=False,
#                             )
#                             full_logits = outputs.logits  # [batch, total_len, vocab_size]
#                             # 仅取 target 段 logits
#                             logits = full_logits[:, -tgt_input_ids.size(1):, :].detach().cpu()
#
#                     elif "marian" in model_type or "opus" in model_type or "t5" in model_type:
#                         # ============================================================
#                         # ✅ Seq2Seq 型 (如 Opus-MT / MarianMT / T5)
#                         # ============================================================
#                         with torch.cuda.amp.autocast(dtype=torch.float16 if device.type == "cuda" else torch.float32):
#                             outputs = model(
#                                 input_ids=src_input_ids,
#                                 attention_mask=src_attention_mask,
#                                 labels=tgt_input_ids,
#                                 output_hidden_states=False,
#                                 output_attentions=False,
#                             )
#                             logits = outputs.logits.detach().cpu()  # [batch, tgt_len, vocab_size]
#                     else:
#                         raise ValueError(f"❌ 未知模型类型: {model_type}")
#                     # ============================================================
#
#                     # 释放显存
#                     del outputs
#                     torch.cuda.empty_cache()
#                     gc.collect()
#
#                     for i in range(len(batch["id"])):
#                         output_data.append({
#                             "id": batch["id"][i].item(),
#                             "src_text": batch["src_text"][i],
#                             "tgt_text": batch["tgt_text"][i],
#                             "src_input_ids": batch["src_input_ids"][i].cpu(),
#                             "src_attention_mask": batch["src_attention_mask"][i].cpu(),
#                             "tgt_input_ids": batch["tgt_input_ids"][i].cpu(),
#                             "tgt_attention_mask": batch["tgt_attention_mask"][i].cpu(),
#                             "task_id": batch["task_id"][i].item(),
#                             "logits": logits[i]
#                         })
#                     success_count += len(batch["id"])
#
#                     # 每 10 批次保存一次，防止过大
#                     if (batch_idx + 1) % 10 == 0:
#                         torch.save(output_data, output_file)
#                         output_data.clear()
#                         gc.collect()
#                         torch.cuda.empty_cache()
#                         logging.info(f"💾 临时保存 {output_file} (到第 {batch_idx + 1} 批次)")
#
#                     # ⏱️ 超时检测 watchdog
#                     elapsed = time.time() - batch_start
#                     if elapsed > 120:
#                         logging.warning(f"⚠️ Batch {batch_idx} 超时 {elapsed:.1f}s，强制清理CUDA上下文")
#                         torch.cuda.empty_cache()
#                         gc.collect()
#
#                 except torch.cuda.OutOfMemoryError:
#                     logging.error(f"💥 CUDA OOM at batch {batch_idx}! 自动回退 batch_size...")
#                     torch.cuda.empty_cache()
#                     if args.batch_size > 1:
#                         args.batch_size = max(1, args.batch_size // 2)
#                         logging.warning(f"⚙️ 新 batch_size={args.batch_size}，重新构建 DataLoader")
#                         shard_dataloader = DataLoader(
#                             TranslationDataset(
#                                 shard_dataset, tokenizer,
#                                 max_seq_len=args.max_seq_len,
#                                 src_lang=src_lang, tgt_lang=tgt_lang, task_id=task_id
#                             ),
#                             batch_size=args.batch_size,
#                             shuffle=False,
#                             num_workers=0,
#                             pin_memory=(device == "cuda"),
#                             collate_fn=lambda b: custom_collate_fn(
#                                 b, max_seq_len=args.max_seq_len, pad_token_id=tokenizer.pad_token_id
#                             )
#                         )
#                         break
#                     else:
#                         logging.error("❌ batch_size=1 仍显存不足，跳过该分片。")
#                         break
#
#                 except RuntimeError as e:
#                     logging.error(f"⚠️ RuntimeError (可能死锁或驱动错误): {e}")
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     time.sleep(2)
#                     continue
#
#                 except Exception as e:
#                     logging.error(f"❌ Batch {batch_idx} 处理失败: {e}")
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     continue
#
#         # ===== 6. 保存结果并验证 =====
#         if len(output_data) > 0:
#             torch.save(output_data, output_file)
#         logging.info(f"💾 保存分片: {output_file}")
#         validate_logits_file(output_file)
#         del output_data
#         gc.collect()
#         torch.cuda.empty_cache()
#
#     # ===== 7. 清理 =====
#     if model is not None:
#         del model
#         gc.collect()
#         torch.cuda.empty_cache()
#
#     logging.info(f"🎉 完成！成功: {success_count} 失败: {fail_count}")
#     return success_count, fail_count
# ==================== 生成 Teacher Logits (基于 Opus-MT 教师模型) ====================
def generate_teacher_logits(args):
    """
    ✅ Opus-MT 教师模型生成 soft logits
    支持 zh→en 和 en→zh 两个方向
    自动切换模型路径 + 显存防护 + 半精度支持
    """
    process = psutil.Process()
    logging.info(f"初始内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    device = torch.device(args.device)

    # ===== 1. 加载数据集 =====
    try:
        dataset_path = os.path.join(RAW_DATA_PATH, "train/*.parquet")
        dataset = load_dataset("parquet", data_files={"train": dataset_path})["train"]
        total_samples = len(dataset)
        logging.info(f"✅ 加载 WMT19 数据集，样本数: {total_samples}")
    except Exception as e:
        logging.error(f"❌ 数据集加载失败: {e}")
        raise

    total_samples = min(args.max_samples or total_samples, total_samples)

    success_count, fail_count = 0, 0

    # ===== 2. 遍历两个翻译方向 =====
    for src_lang, tgt_lang, task_id, output_prefix, model_path in [
        ("zh", "en", 0, os.path.join(TEACHER_LOGITS_DIR, "zh_to_en"), OPUS_MT_ZH_EN),
        ("en", "zh", 1, os.path.join(TEACHER_LOGITS_DIR, "en_to_zh"), OPUS_MT_EN_ZH),
    ]:
        logging.info(f"🧠 生成 {src_lang}→{tgt_lang} logits (任务ID={task_id})")

        model = None
        tokenizer = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                local_files_only=True
            ).to(device)
            model.eval()
            logging.info(f"✅ 成功加载 Opus 模型 ({src_lang}→{tgt_lang})")
        except Exception as e:
            logging.error(f"❌ 教师模型加载失败 ({src_lang}->{tgt_lang}): {e}")
            continue

        if model is None or tokenizer is None:
            logging.error(f"⚠️ 未能初始化模型或 tokenizer ({src_lang}->{tgt_lang})，跳过此方向。")
            continue

        # ===== 2.2 选取分片 =====
        start_idx = args.start_from
        end_idx = min(start_idx + args.shard_size, total_samples)
        shard_dataset = dataset.select(range(start_idx, end_idx))
        if len(shard_dataset) == 0:
            logging.warning(f"⚠️ 分片 {args.shard_idx} 无样本，跳过。")
            continue

        pad_id = tokenizer.pad_token_id

        shard_dataloader = DataLoader(
            TranslationDataset(
                shard_dataset, tokenizer,
                max_seq_len=args.max_seq_len,
                src_lang=src_lang, tgt_lang=tgt_lang, task_id=task_id
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
            collate_fn=lambda b, pad_id=pad_id: custom_collate_fn(
                b, max_seq_len=args.max_seq_len, pad_token_id=pad_id
            )
        )

        output_file = f"{output_prefix}_shard_{args.shard_idx}.pt"
        output_data = []

        # ===== 3. 主生成循环 =====
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(shard_dataloader, desc=f"{src_lang}→{tgt_lang} 分片 {args.shard_idx}")
            ):
                batch_start = time.time()
                try:
                    src_input_ids = batch["src_input_ids"].to(device)
                    src_attention_mask = batch["src_attention_mask"].to(device)
                    tgt_input_ids = batch["tgt_input_ids"].to(device)

                    # 🔹 Opus-MT / MarianMT 是标准 encoder-decoder 模型
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
                        outputs = model(
                            input_ids=src_input_ids,
                            attention_mask=src_attention_mask,
                            labels=tgt_input_ids,
                            output_hidden_states=False,
                            output_attentions=False,
                        )
                        logits = outputs.logits.detach().cpu()

                    for i in range(len(batch["id"])):
                        output_data.append({
                            "id": batch["id"][i].item(),
                            "src_text": batch["src_text"][i],
                            "tgt_text": batch["tgt_text"][i],
                            "src_input_ids": batch["src_input_ids"][i].cpu(),
                            "src_attention_mask": batch["src_attention_mask"][i].cpu(),
                            "tgt_input_ids": batch["tgt_input_ids"][i].cpu(),
                            "tgt_attention_mask": batch["tgt_attention_mask"][i].cpu(),
                            "task_id": batch["task_id"][i].item(),
                            "logits": logits[i]
                        })
                    success_count += len(batch["id"])

                    # 每10批次中间保存一次
                    if (batch_idx + 1) % 10 == 0:
                        torch.save(output_data, output_file)
                        output_data.clear()
                        torch.cuda.empty_cache()
                        gc.collect()
                        logging.info(f"💾 临时保存 {output_file} (到第 {batch_idx + 1} 批次)")

                    # 超时 watchdog
                    elapsed = time.time() - batch_start
                    if elapsed > 120:
                        logging.warning(f"⚠️ Batch {batch_idx} 超时 {elapsed:.1f}s，强制清理CUDA上下文")
                        torch.cuda.empty_cache()
                        gc.collect()

                except torch.cuda.OutOfMemoryError:
                    logging.error(f"💥 CUDA OOM at batch {batch_idx}! 自动回退 batch_size...")
                    torch.cuda.empty_cache()
                    if args.batch_size > 1:
                        args.batch_size = max(1, args.batch_size // 2)
                        break
                    else:
                        logging.error("❌ batch_size=1 仍显存不足，跳过该分片。")
                        break

                except Exception as e:
                    logging.error(f"❌ Batch {batch_idx} 处理失败: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

        # ===== 4. 保存分片结果 =====
        if len(output_data) > 0:
            torch.save(output_data, output_file)
        logging.info(f"💾 保存分片: {output_file}")
        validate_logits_file(output_file)
        del output_data
        gc.collect()
        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"🎉 生成完成！成功: {success_count} 失败: {fail_count}")
    return success_count, fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Teacher Logits (优化版)")
    parser.add_argument("--dataset_path", type=str, default="data/raw/wmt19_zh_en", help="WMT19路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--max_seq_len", type=int, default=ModelConfig.MAX_SEQ_LEN, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--start_from", type=int, default=0, help="起始索引")
    parser.add_argument("--shard_size", type=int, default=100000, help="分片大小")
    parser.add_argument("--compile", action="store_true", help="torch.compile")
    parser.add_argument("--int8", action="store_true", help="INT8量化")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_api", action="store_true", help="使用API")
    parser.add_argument("--simulate_quant_noise", action="store_true",
                        help="是否在生成 logits 时加入模拟量化误差（增强学生鲁棒性）")
    parser.add_argument("--noise_std", type=float, default=0.01,
                        help="模拟量化噪声标准差 (默认 0.01)")
    parser.add_argument("--shard_idx", type=int, default=0, help="当前分片索引（用于命名）")
    args = parser.parse_args()

    try:
        success, fail = generate_teacher_logits(args)
        logging.info(f"🎉 处理完成！成功: {success}, 失败: {fail}")
    except Exception as e:
        logging.error(f"❌ 程序异常: {e}")
        raise
