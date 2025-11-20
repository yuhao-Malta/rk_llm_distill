import os
import sys
import gc
import psutil
import time
import torch
import logging
import sacrebleu
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_seq2seq_transformer import TinySeq2SeqTransformer
from config.config import ModelConfig, EvalConfig, MODEL_PATH, OPUS_MT_ZH_EN, OPUS_MT_EN_ZH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# ===================================================================
#   加载任意模型：HF model / Tiny Student model
# ===================================================================
def load_any_model(model_path, device="cpu", is_student=False):
    try:
        if is_student:
            logging.info("🧠 加载蒸馏学生模型 (TinySeq2SeqTransformer, 双向) ...")

            # ✅ 学生模型从教师 tokenizer 动态加载
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

            # ✅ 加载学生模型权重
            state_dict = torch.load(model_path, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"⚠️ Missing keys: {missing}")
            if unexpected:
                logging.warning(f"⚠️ Unexpected keys: {unexpected}")

            model.eval()
            logging.info("✅ 学生模型加载成功")
            return model

        # 试图加载 HF seq2seq
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, device_map=device, trust_remote_code=True
            )
            logging.info("✅ HF Seq2Seq 模型加载成功")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device, trust_remote_code=True
            )
            logging.info("✅ HF CausalLM 模型加载成功")

        model.eval()
        return model

    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        raise


# ===================================================================
#   载入评估集
# ===================================================================
def load_test_dataset(test_data_path, tasks=None, max_samples=100):
    tasks = tasks or EvalConfig.TASKS
    datasets = {}

    dataset = load_dataset("parquet", data_files={"test": test_data_path})["test"]
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    if "translation" not in dataset.column_names:
        raise ValueError(f"❌ 数据集格式错误: {dataset.column_names}")

    for task in tasks:
        src_lang, tgt_lang = task.split("_to_")
        task_dataset = []

        for item in dataset:
            if src_lang in item["translation"] and tgt_lang in item["translation"]:
                task_dataset.append({
                    "translation": {
                        src_lang: item["translation"][src_lang],
                        tgt_lang: item["translation"][tgt_lang],
                    }
                })

        datasets[task] = task_dataset
        logging.info(f"📚 {task}: {len(task_dataset)} 条样本")

    return datasets


# ===================================================================
# 翻译函数 (兼容 Tiny 双向学生模型)
# ===================================================================
def translate_with_student(model, text, task, device, max_len):
    task_id = 0 if task == "zh_to_en" else 1

    tokenizer = model.tokenizer_zh2en if task_id == 0 else model.tokenizer_en2zh

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id or eos

    pred_ids = model.generate(
        input_ids=encoded["input_ids"],
        max_length=max_len,
        task_id=task_id,
        num_beams=4,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )

    return tokenizer.decode(pred_ids[0], skip_special_tokens=True)


def translate_texts(model, texts, task, device, max_len):
    return [translate_with_student(model, text, task, device, max_len) for text in texts]


# ===================================================================
# BLEU 计算
# ===================================================================
def compute_bleu(model, dataset, task, device, max_len):
    src_lang, tgt_lang = task.split("_to_")

    src_texts = [item["translation"][src_lang] for item in dataset]
    ref_texts = [item["translation"][tgt_lang] for item in dataset]

    hyps = translate_texts(model, src_texts, task, device, max_len)

    bleu = sacrebleu.corpus_bleu(hyps, [ref_texts])
    return bleu.score


# ===================================================================
# 推理延迟
# ===================================================================
def measure_inference_latency(model, dataset, task, device, max_len, num_samples=50):
    src_lang = task.split("_to_")[0]
    src_texts = [item["translation"][src_lang] for item in dataset[:num_samples]]

    times = []

    for text in src_texts:
        start = time.time()
        _ = translate_with_student(model, text, task, device, max_len)
        times.append(time.time() - start)

    return sum(times) / len(times) * 1000


# ===================================================================
# 主评估入口
# ===================================================================
def main(model_path, is_student=False, tasks=None, max_samples=None, test_data_path=None, device="cpu"):
    tasks = tasks or EvalConfig.TASKS
    max_samples = max_samples or EvalConfig.MAX_EVAL_SAMPLES
    test_data_path = test_data_path or EvalConfig.TEST_DATA_PATH

    logging.info("=" * 60)
    logging.info("🚀 开始模型评估")
    logging.info("=" * 60)

    model = load_any_model(model_path, device=device, is_student=is_student)
    datasets = load_test_dataset(test_data_path, tasks, max_samples)
    mem_usage = get_memory_usage()

    results = {}

    for task in tasks:
        logging.info(f"\n🧪 评估任务: {task}")
        bleu_score = compute_bleu(model, datasets[task], task, device, ModelConfig.MAX_SEQ_LEN)
        latency = measure_inference_latency(model, datasets[task], task, device, ModelConfig.MAX_SEQ_LEN)

        results[task] = {"bleu": bleu_score, "latency_ms": latency}
        logging.info(f"✅ {task}: BLEU={bleu_score:.2f}, 延迟={latency:.2f}ms")

    logging.info(f"\n📊 内存占用: {mem_usage:.2f} MB")

    for task, m in results.items():
        logging.info(f"{task}: BLEU={m['bleu']:.2f}, 延迟={m['latency_ms']:.2f}ms")

    return results


# ===================================================================
# CLI 入口
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估蒸馏模型（中英互译）")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--is_student", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_samples", type=int, default=EvalConfig.MAX_EVAL_SAMPLES)
    parser.add_argument("--test_data_path", type=str, default=EvalConfig.TEST_DATA_PATH)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        is_student=args.is_student,
        max_samples=args.max_samples,
        test_data_path=args.test_data_path,
        device=args.device,
    )

