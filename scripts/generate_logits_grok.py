import os
import torch
import psutil  # 添加内存监控
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import logging
import time

try:
    import dashscope
except ImportError:
    dashscope = None

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 提高日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generate_logits.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置DashScope API Key（替换为您的真实Key）
DASHSCOPE_API_KEY = "sk-b0c78a77e5ea489b8c68e0b5049204c6"  # 请替换！


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
def call_qwen_translate_api(text, target_lang="en", max_retries=3):
    if not dashscope:
        raise ImportError("❌ DashScope未安装，请运行 'pip install dashscope'")
    dashscope.api_key = DASHSCOPE_API_KEY
    TEXT_TRANSLATION_AVAILABLE = hasattr(dashscope, 'TextTranslation')

    for attempt in range(max_retries):
        logging.info(f"API 翻译尝试 {attempt + 1}/{max_retries}")
        try:
            if TEXT_TRANSLATION_AVAILABLE:
                response = dashscope.TextTranslation.call(
                    model='qwen-max',
                    text=text,
                    target_language=target_lang
                )
                if response.status_code == 200:
                    return response.output['translated_text']
                logging.warning(f"⚠️ TextTranslation失败: {response.message}")
            else:
                source_lang_full = "中文" if target_lang == "en" else "英文"
                target_lang_full = "英文" if target_lang == "en" else "中文"
                prompt = (
                    f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
                    f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
                    f"只输出翻译结果：\n\n{text}"
                )
                response = dashscope.Generation.call(
                    model='qwen-max',
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.1
                )
                if response.status_code == 200:
                    return response.output.get('text', '').strip()
                logging.warning(f"⚠️ Generation失败: {response.message}")
        except Exception as e:
            logging.warning(f"⚠️ API调用异常: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    logging.error(f"❌ API翻译失败（{max_retries}次尝试）")
    return None


# 本地模型翻译
def call_qwen_translate_local(model, tokenizer, text, target_lang="en", max_seq_len=64):
    source_lang_full = "中文" if target_lang == "en" else "英文"
    target_lang_full = "英文" if target_lang == "en" else "中文"
    prompt = (
        f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
        f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
        f"只输出翻译结果：\n\n{text}"
    )
    inputs = tokenizer(
        prompt, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
    ).to(model.device)
    inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
    inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits  # 返回logits

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len=64, src_lang="zh", tgt_lang="en"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        source_lang_full = "中文" if self.src_lang == "zh" else "英文"
        target_lang_full = "英文" if self.tgt_lang == "en" else "中文"
        prompt = (
            f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
            f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
            f"只输出翻译结果：\n\n{src_text}"
        )

        src_encoding = self.tokenizer(
            prompt, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        tgt_encoding = self.tokenizer(
            tgt_text, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "idx": idx,
            "src_input_ids": src_encoding["input_ids"].squeeze().to(dtype=torch.long),
            "src_attention_mask": src_encoding["attention_mask"].squeeze().to(dtype=torch.long),
            "tgt_input_ids": tgt_encoding["input_ids"].squeeze().to(dtype=torch.long),
            "tgt_attention_mask": tgt_encoding["attention_mask"].squeeze().to(dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

# 自定义 collate_fn
def custom_collate_fn(batch, max_seq_len=64, pad_token_id=151643):
    src_input_ids = [item["src_input_ids"] for item in batch]
    src_attention_mask = [item["src_attention_mask"] for item in batch]
    tgt_input_ids = [item["tgt_input_ids"] for item in batch]
    tgt_attention_mask = [item["tgt_attention_mask"] for item in batch]
    idx = [item["idx"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    src_input_ids = torch.nn.utils.rnn.pad_sequence(
        src_input_ids, batch_first=True, padding_value=pad_token_id
    )[:, :max_seq_len].to(dtype=torch.long)
    src_attention_mask = torch.nn.utils.rnn.pad_sequence(
        src_attention_mask, batch_first=True, padding_value=0
    )[:, :max_seq_len].to(dtype=torch.long)
    tgt_input_ids = torch.nn.utils.rnn.pad_sequence(
        tgt_input_ids, batch_first=True, padding_value=pad_token_id
    )[:, :max_seq_len].to(dtype=torch.long)
    tgt_attention_mask = torch.nn.utils.rnn.pad_sequence(
        tgt_attention_mask, batch_first=True, padding_value=0
    )[:, :max_seq_len].to(dtype=torch.long)
    idx = torch.tensor(idx, dtype=torch.long)

    return {
        "idx": idx,
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_mask,
        "tgt_input_ids": tgt_input_ids,
        "tgt_attention_mask": tgt_attention_mask,
        "src_text": src_texts,
        "tgt_text": tgt_texts
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
                assert isinstance(item["src"], str) and isinstance(item["ref"], str) and isinstance(item["hyp"], str), f"src/ref/hyp类型错误: {item}"
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
                            logging.warning(f"⚠️ logits维度不匹配 (预期: {vocab_size}, 实际: {item['logits'].shape[-1]})")
                        else:
                            logging.debug(f"✅ logits维度验证通过: {item['logits'].shape}")
                    else:
                        logging.debug(f"⚠️ 未提供model_path，跳过logits维度检查，实际维度: {item['logits'].shape}")
            logging.info(f"✅ 验证通过！{file_path} 共 {count} 条有效记录（PT，{'API模式' if use_api else '本地模式'}）")
        return count
    except Exception as e:
        logging.error(f"❌ 验证失败: {e}")
        return 0

# 生成 teacher logits
def generate_teacher_logits(args):
    """
    生成QWen-1.5-1.8B软标签（中→英，英→中），保存为.pt文件
    优化：支持本地/API、批处理、FP16/INT8、max_seq_len=64
    """
    # 内存监控
    process = psutil.Process()
    logging.debug(
        f"初始内存使用: {process.memory_info().rss / 1024 ** 2:.2f} MB, 可用系统内存: {psutil.virtual_memory().available / 1024 ** 2:.2f} MB")

    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        logging.info("✅ 成功加载 Qwen Tokenizer")
        logging.debug(f"加载tokenizer后内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    except Exception as e:
        logging.error(f"❌ 加载tokenizer失败: {e}")
        raise

    # 加载模型（本地或API）
    model = None
    if not args.use_api:
        try:
            check_model_files(MODEL_PATH)
            logging.debug("开始加载 Qwen-1.5-1.8B 模型...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map=args.device,
                torch_dtype=torch.float32,  # 使用 FP32 避免内存溢出
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            if args.int8:
                logging.debug("应用INT8量化...")
                model = torch.quantization.quantize_dynamic(
                    model,
                    qconfig_spec={
                        torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                        torch.nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig
                    },
                    dtype=torch.qint8
                )
            model.eval()
            if args.compile and args.device == "cuda":
                model = torch.compile(model)
            device = next(model.parameters()).device
            logging.info(f"✅ 成功加载 Qwen-1.5-1.8B 到 {device}")
            logging.debug(f"加载模型后内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        except Exception as e:
            logging.error(f"❌ 加载本地模型失败: {e}")
            if args.use_api:
                logging.warning("⚠️ 回退到DashScope API")
            else:
                raise
    else:
        device = "cpu"

    # 加载WMT19数据集
    dataset_path = os.path.join(PROJECT_ROOT, args.dataset_path)
    try:
        dataset = load_dataset("parquet", data_files={"train": f"{dataset_path}/train/*.parquet"})["train"]
        if args.debug:
            dataset = dataset.select(range(1))
        logging.info(f"✅ 加载WMT19数据集，样本数: {len(dataset)}")
        logging.debug(f"加载数据集后内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    except Exception as e:
        logging.error(f"❌ 加载数据集失败: {e}")
        raise

    # 设置样本范围和分片
    total_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    shard_size = min(args.shard_size, total_samples)  # 确保分片大小合理
    num_shards = (total_samples + shard_size - 1) // shard_size

    # 生成中→英和英→中
    success_count = 0
    for src_lang, tgt_lang, task_id, output_prefix in [
        ("zh", "en", 0, os.path.join(OUTPUT_DIR, "zh_to_en")),
        ("en", "zh", 1, os.path.join(OUTPUT_DIR, "en_to_zh"))
    ]:
        logging.info(f"🧠 生成 {src_lang}→{tgt_lang} 软标签，分片大小: {shard_size}")
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, total_samples)
            shard_dataset = dataset.select(range(args.start_from + start_idx, args.start_from + end_idx))
            shard_dataloader = DataLoader(
                TranslationDataset(shard_dataset, tokenizer, max_seq_len=args.max_seq_len, src_lang=src_lang,
                                   tgt_lang=tgt_lang),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(args.device == "cuda"),
                collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=args.max_seq_len, pad_token_id=tokenizer.pad_token_id)
            )

            output_file = f"{output_prefix}_shard_{shard_idx}.{'jsonl' if args.use_api else 'pt'}"
            output_data = []

            if args.use_api:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for batch_idx, batch in enumerate(
                            tqdm(shard_dataloader, desc=f"{src_lang}→{tgt_lang} 分片 {shard_idx}")):
                        start_time = time.time()
                        logging.debug(f"批次 {batch_idx + 1} 开始，内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")

                        idx = batch["idx"]
                        src_texts = batch["src_text"]
                        tgt_texts = batch["tgt_text"]

                        try:
                            for i in range(len(idx)):
                                trans_text = call_qwen_translate_api(src_texts[i], tgt_lang)
                                if trans_text:
                                    item = {
                                        "id": idx[i].item(),
                                        "src": src_texts[i],
                                        "ref": tgt_texts[i],
                                        "hyp": trans_text,
                                        "task_id": task_id,
                                        "timestamp": time.time()
                                    }
                                    json.dump(item, f, ensure_ascii=False)
                                    f.write('\n')
                                    output_data.append(item)
                                    success_count += 1
                                else:
                                    logging.warning(f"⚠️ API翻译失败: ID {idx[i].item()}")
                            logging.info(
                                f"批次 {batch_idx + 1}/{len(shard_dataloader)} 完成，耗时: {time.time() - start_time:.2f}s, 内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
                        except Exception as e:
                            logging.warning(f"⚠️ 批次 {batch_idx + 1} 处理失败: {e}")
            else:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(
                            tqdm(shard_dataloader, desc=f"{src_lang}→{tgt_lang} 分片 {shard_idx}")):
                        start_time = time.time()
                        logging.debug(f"批次 {batch_idx + 1} 开始，内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")

                        idx = batch["idx"]
                        src_texts = batch["src_text"]
                        tgt_texts = batch["tgt_text"]
                        src_input_ids = batch["src_input_ids"].to(device).to(dtype=torch.long)
                        src_attention_mask = batch["src_attention_mask"].to(device).to(dtype=torch.long)
                        try:
                            outputs = model(input_ids=src_input_ids, attention_mask=src_attention_mask)
                            logits = outputs.logits  # [batch, seq_len, vocab_size]
                            for i in range(len(idx)):
                                output_data.append({
                                    "id": idx[i].item(),
                                    "src_input_ids": src_input_ids[i].cpu().to(dtype=torch.long),
                                    "src_attention_mask": src_attention_mask[i].cpu().to(dtype=torch.long),
                                    "tgt_input_ids": batch["tgt_input_ids"][i].cpu().to(dtype=torch.long),
                                    "tgt_attention_mask": batch["tgt_attention_mask"][i].cpu().to(dtype=torch.long),
                                    "logits": logits[i].cpu(),
                                    "task_id": task_id,
                                    "src_text": src_texts[i],
                                    "tgt_text": tgt_texts[i],
                                    "hyp_text": None
                                })
                            success_count += len(idx)
                            logging.info(
                                f"批次 {batch_idx + 1}/{len(shard_dataloader)} 完成，耗时: {time.time() - start_time:.2f}s, 内存: {process.memory_info().rss / 1024 ** 2:.2f} MB")
                        except Exception as e:
                            logging.warning(f"⚠️ 批次 {batch_idx + 1} 处理失败: {e}")

                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                        # 释放 CPU 内存
                        torch.cuda.empty_cache() if args.device == "cuda" else torch.cpu.empty_cache()

                    torch.save(output_data, output_file)

            logging.info(f"✅ {src_lang}→{tgt_lang} 分片 {shard_idx} 完成！成功: {success_count}, 保存到 {output_file}")
            validate_logits_file(output_file, use_api=args.use_api, model_path=MODEL_PATH)
            # 释放 output_data
            output_data = []
            import gc
            gc.collect()

    # 释放模型内存
    if model is not None:
        del model
        torch.cuda.empty_cache() if args.device == "cuda" else torch.cpu.empty_cache()
        gc.collect()

    return success_count, 0

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 QWen-1.5-1.8B 软标签")
    parser.add_argument("--dataset_path", type=str, default="data/raw/wmt19_zh_en", help="WMT19 数据集路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_seq_len", type=int, default=64, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（None=全量）")
    parser.add_argument("--start_from", type=int, default=0, help="开始样本索引")
    parser.add_argument("--shard_size", type=int, default=100000, help="每片样本数（本地模式）")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile 加速")
    parser.add_argument("--int8", action="store_true", help="使用 INT8 量化（CPU）")
    parser.add_argument("--debug", action="store_true", help="调试模式（1 条）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--use_api", action="store_true", help="使用 DashScope API（生成 jsonl）")
    args = parser.parse_args()

    try:
        success, fail = generate_teacher_logits(args)
        logging.info(f"🎉 处理完成！成功: {success}, 失败: {fail}")
    except Exception as e:
        logging.error(f"❌ 主程序异常退出: {e}")
        raise
