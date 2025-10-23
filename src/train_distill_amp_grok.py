import sys
import os

from torch import GradScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast  # 更新为 torch.amp.autocast
from tqdm import tqdm
import logging
from models.tiny_transformer import TinyTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train_distill_amp.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logits", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# 1. 自定义数据集类（支持.pt和.jsonl）
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, file_path, max_samples=None, is_jsonl=True, max_seq_len=64, tokenizer=None):
        self.file_path = file_path
        self.max_samples = max_samples
        self.is_jsonl = is_jsonl
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data = []

        if is_jsonl:
            logging.info(f"加载并分词 {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    item = json.loads(line)
                    src_encoded = tokenizer(item["src"], return_tensors="pt", truncation=True, max_length=max_seq_len)
                    tgt_encoded = tokenizer(item["hyp"], return_tensors="pt", truncation=True, max_length=max_seq_len)
                    self.data.append({
                        "src_input_ids": src_encoded.input_ids.squeeze(0).to(dtype=torch.long),
                        "src_attention_mask": src_encoded.attention_mask.squeeze(0).to(dtype=torch.long),
                        "tgt_input_ids": tgt_encoded.input_ids.squeeze(0).to(dtype=torch.long),
                        "tgt_attention_mask": tgt_encoded.attention_mask.squeeze(0).to(dtype=torch.long),
                        "task_id": item["task_id"]
                    })
        else:
            logging.info(f"加载 .pt 文件: {file_path}")
            data = torch.load(file_path)
            self.data = data[:max_samples] if max_samples else data
            for item in self.data:
                assert "logits" in item, f"缺少 teacher_logits: {item}"
                assert item["logits"].shape == (max_seq_len, 151936), f"无效 logits 形状: {item['logits'].shape}"
                assert item.get("src_input_ids", item.get("input_ids")) is not None, f"缺少 src_input_ids 或 input_ids: {item}"
                assert item.get("tgt_input_ids", item.get("hyp_input_ids", item.get("input_ids"))) is not None, f"缺少 tgt_input_ids, hyp_input_ids 或 input_ids: {item}"
                # 确保数据类型
                item["src_input_ids"] = item.get("src_input_ids", item.get("input_ids")).to(dtype=torch.long)
                item["src_attention_mask"] = item.get("src_attention_mask", item.get("attention_mask")).to(dtype=torch.long)
                item["tgt_input_ids"] = item.get("tgt_input_ids", item.get("hyp_input_ids", item.get("input_ids"))).to(dtype=torch.long)
                item["tgt_attention_mask"] = item.get("tgt_attention_mask", item.get("hyp_attention_mask", item.get("attention_mask"))).to(dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -----------------------------
# 2. 自定义 collate_fn（批次填充）
# -----------------------------
def custom_collate_fn(batch, max_seq_len=64, pad_token_id=151643):
    src_input_ids = [item.get("src_input_ids", item.get("input_ids")) for item in batch]
    src_attention_mask = [item.get("src_attention_mask", item.get("attention_mask")) for item in batch]
    tgt_input_ids = [item.get("tgt_input_ids", item.get("hyp_input_ids", item.get("input_ids"))) for item in batch]
    tgt_attention_mask = [item.get("tgt_attention_mask", item.get("hyp_attention_mask", item.get("attention_mask"))) for item in batch]
    task_ids = [item["task_id"] for item in batch]
    teacher_logits = [item.get("logits") for item in batch] if not any("logits" not in item for item in batch) else None

    if any(x is None for x in src_input_ids + src_attention_mask + tgt_input_ids + tgt_attention_mask):
        raise KeyError("缺少必要的输入字段：src_input_ids, src_attention_mask, tgt_input_ids, 或 tgt_attention_mask")

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
    task_ids = torch.tensor(task_ids, dtype=torch.long)

    batch_dict = {
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_mask,
        "tgt_input_ids": tgt_input_ids,
        "tgt_attention_mask": tgt_attention_mask,
        "task_id": task_ids
    }
    if teacher_logits:
        batch_dict["teacher_logits"] = torch.stack(teacher_logits)
    return batch_dict

# -----------------------------
# 3. KL 散度损失函数
# -----------------------------
def kl_div_loss(student_logits, teacher_logits, temperature=2.0, vocab_size=151936):
    seq_len_s = student_logits.size(1)
    seq_len_t = teacher_logits.size(1)
    min_len = min(seq_len_s, seq_len_t)
    if seq_len_s != seq_len_t:
        logging.warning(f"⚠️ 序列长度不匹配 ({seq_len_s} vs {seq_len_t})，同步到 {min_len}")
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]

    teacher_vocab_size = teacher_logits.size(-1)
    if teacher_vocab_size != vocab_size:
        logging.warning(f"⚠️ 词汇表大小不匹配 (教师: {teacher_vocab_size} vs 学生: {vocab_size})，裁剪到 {vocab_size}")
        teacher_logits = teacher_logits[:, :, :vocab_size]

    s_dist = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    t_dist = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return nn.functional.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)

# -----------------------------
# 4. 预计算 teacher_logits
# -----------------------------
def precompute_teacher_logits(teacher_model, dataloader, cache_file, device, max_seq_len=64):
    if os.path.exists(cache_file):
        logging.info(f"✅ 找到缓存: {cache_file}")
        return torch.load(cache_file)

    logging.info(f"🧠 预计算 teacher_logits，保存至: {cache_file}")
    teacher_model.eval()
    cached_data = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预计算 teacher_logits"):
            input_ids = batch["tgt_input_ids"].to(device).to(dtype=torch.long)
            attention_mask = batch["tgt_attention_mask"].to(device).to(dtype=torch.long)
            try:
                logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
                for i in range(len(batch["src_input_ids"])):
                    cached_data.append({
                        "src_input_ids": batch["src_input_ids"][i].cpu().to(dtype=torch.long),
                        "src_attention_mask": batch["src_attention_mask"][i].cpu().to(dtype=torch.long),
                        "tgt_input_ids": batch["tgt_input_ids"][i].cpu().to(dtype=torch.long),
                        "tgt_attention_mask": batch["tgt_attention_mask"][i].cpu().to(dtype=torch.long),
                        "task_id": batch["task_id"][i].cpu(),
                        "logits": logits[i]
                    })
            except Exception as e:
                logging.error(f"❌ 预计算失败: {e}")
                raise

    torch.save(cached_data, cache_file)
    logging.info(f"💾 teacher_logits 已缓存至: {cache_file}")
    return cached_data

# -----------------------------
# 5. 蒸馏训练主函数
# -----------------------------
def train_distill_amp(
        teacher_logits_dir="data/teacher_logits",
        output_model_dir="outputs/models",
        epochs=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        max_samples_per_task=None,
        device=None,
        use_jsonl=False,
        teacher_model_path=MODEL_PATH,
        max_seq_len=64,
        compile=False,
        shard_idx=0,
        patience=2
):
    logging.info("🚀 开始多任务蒸馏训练 (AMP + 大 Batch)...")
    logging.info(f"📦 使用数据格式: {'jsonl' if use_jsonl else 'pt'}")

    # 1. 设置设备
    if isinstance(device, str):
        device = torch.device(device.lower() if torch.cuda.is_available() and device.lower() == "cuda" else "cpu")
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"📦 使用设备: {device}")

    # 2. 创建输出目录
    os.makedirs(output_model_dir, exist_ok=True)

    # 3. 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_path, local_files_only=True, trust_remote_code=True
        )
        config_path = os.path.join(teacher_model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        vocab_size = config.get("vocab_size", len(tokenizer))
        logging.info(f"✅ Qwen Tokenizer 词汇表大小: {len(tokenizer)}，模型 vocab_size: {vocab_size}")
    except Exception as e:
        logging.error(f"❌ 加载 Qwen Tokenizer 或 config.json 失败: {e}")
        raise

    # 4. 初始化学生模型
    try:
        model = TinyTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=128,
            nhead=4,
            num_layers=2,
            share_weights=True
        ).to(device)
        model.tokenizer = tokenizer
        if compile and device.type == "cuda":
            model = torch.compile(model)
        logging.info(f"✅ 学生模型 TinyTransformer 初始化完成，参数量: {model.num_parameters()/1e6:.1f}M")
    except Exception as e:
        logging.error(f"❌ 初始化 TinyTransformer 失败: {e}")
        raise

    # 5. 加载教师模型（仅在 use_jsonl=True 且无缓存时需要）
    teacher_model = None
    if use_jsonl:
        try:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path, device_map=device, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, trust_remote_code=True
            ).eval()
            if compile and device.type == "cuda":
                teacher_model = torch.compile(teacher_model)
            logging.info(f"✅ 教师模型加载完成: {teacher_model_path}")
        except Exception as e:
            logging.error(f"❌ 加载教师模型失败: {e}")
            raise

    # 6. 加载数据集
    zh_to_en_path = os.path.join(teacher_logits_dir, f"zh_to_en_shard_{shard_idx}.{'jsonl' if use_jsonl else 'pt'}")
    en_to_zh_path = os.path.join(teacher_logits_dir, f"en_to_zh_shard_{shard_idx}.{'jsonl' if use_jsonl else 'pt'}")
    logging.info(f"加载中→英数据: {zh_to_en_path}")
    logging.info(f"加载英→中数据: {en_to_zh_path}")

    try:
        if not os.path.exists(zh_to_en_path):
            raise FileNotFoundError(f"❌ 未找到中→英数据: {zh_to_en_path}")
        if not os.path.exists(en_to_zh_path):
            raise FileNotFoundError(f"❌ 未找到英→中数据: {en_to_zh_path}")
        dataset_zh_en = TranslationDataset(zh_to_en_path, max_samples=max_samples_per_task, is_jsonl=use_jsonl, max_seq_len=max_seq_len, tokenizer=tokenizer)
        dataset_en_zh = TranslationDataset(en_to_zh_path, max_samples=max_samples_per_task, is_jsonl=use_jsonl, max_seq_len=max_seq_len, tokenizer=tokenizer)
    except Exception as e:
        logging.error(f"❌ 数据集加载失败: {e}")
        raise

    # 7. 预计算 teacher_logits（仅在 use_jsonl=True 时）
    if use_jsonl:
        zh_to_en_cache = os.path.join(CACHE_DIR, f"zh_to_en_shard_{shard_idx}_cache.pt")
        en_to_zh_cache = os.path.join(CACHE_DIR, f"en_to_zh_shard_{shard_idx}_cache.pt")
        loader_zh_en = DataLoader(
            dataset_zh_en, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
        )
        loader_en_zh = DataLoader(
            dataset_en_zh, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
        )
        if teacher_model:
            dataset_zh_en.data = precompute_teacher_logits(teacher_model, loader_zh_en, zh_to_en_cache, device, max_seq_len)
            dataset_en_zh.data = precompute_teacher_logits(teacher_model, loader_en_zh, en_to_zh_cache, device, max_seq_len)
            loader_zh_en = DataLoader(
                dataset_zh_en, batch_size=batch_size, shuffle=True, num_workers=0,
                pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
            )
            loader_en_zh = DataLoader(
                dataset_en_zh, batch_size=batch_size, shuffle=True, num_workers=0,
                pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
            )
    else:
        loader_zh_en = DataLoader(
            dataset_zh_en, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
        )
        loader_en_zh = DataLoader(
            dataset_en_zh, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), collate_fn=lambda batch: custom_collate_fn(batch, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
        )

    logging.info(f"📊 数据集统计:")
    logging.info(f"  - 中→英样本数: {len(dataset_zh_en)}")
    logging.info(f"  - 英→中样本数: {len(dataset_en_zh)}")

    # 8. 初始化优化器、AMP 缩放器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 9. 训练循环（带 early stopping）
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        logging.info(f"\n📅 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        total_batches = 0

        # 中→英训练
        logging.info("🧠 训练中→英任务...")
        pbar_zh_en = tqdm(loader_zh_en, desc="中→英", leave=False)
        for step, batch in enumerate(pbar_zh_en):
            try:
                input_ids = batch["src_input_ids"].to(device).to(dtype=torch.long)
                attention_mask = batch["src_attention_mask"].to(device).to(dtype=torch.long)
                task_ids = batch["task_id"].to(device)
                tgt_input_ids = batch["tgt_input_ids"].to(device).to(dtype=torch.long)
                tgt_attention_mask = batch["tgt_attention_mask"].to(device).to(dtype=torch.long)

                with torch.no_grad():
                    teacher_logits = batch.get("teacher_logits", None)
                    if teacher_logits is None and teacher_model:
                        teacher_logits = teacher_model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask).logits
                    elif teacher_logits is None:
                        teacher_logits = model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, task_id=task_ids)["logits"]
                    else:
                        teacher_logits = teacher_logits.to(device)

                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_id=task_ids)
                    student_logits = outputs["logits"]
                    logging.debug(f"🔍 student_logits.shape: {student_logits.shape}")
                    logging.debug(f"🔍 teacher_logits.shape: {teacher_logits.shape}")
                    loss = kl_div_loss(student_logits, teacher_logits, vocab_size=vocab_size)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(pbar_zh_en):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                total_batches += 1
                pbar_zh_en.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            except Exception as e:
                logging.error(f"❌ 中→英训练失败 (step {step}): {e}")
                raise

        # 英→中训练
        logging.info("🧠 训练英→中任务...")
        pbar_en_zh = tqdm(loader_en_zh, desc="英→中", leave=False)
        for step, batch in enumerate(pbar_en_zh):
            try:
                input_ids = batch["src_input_ids"].to(device).to(dtype=torch.long)
                attention_mask = batch["src_attention_mask"].to(device).to(dtype=torch.long)
                task_ids = batch["task_id"].to(device)
                tgt_input_ids = batch["tgt_input_ids"].to(device).to(dtype=torch.long)
                tgt_attention_mask = batch["tgt_attention_mask"].to(device).to(dtype=torch.long)

                with torch.no_grad():
                    teacher_logits = batch.get("teacher_logits", None)
                    if teacher_logits is None and teacher_model:
                        teacher_logits = teacher_model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask).logits
                    elif teacher_logits is None:
                        teacher_logits = model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, task_id=task_ids)["logits"]
                    else:
                        teacher_logits = teacher_logits.to(device)

                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_id=task_ids)
                    student_logits = outputs["logits"]
                    logging.debug(f"🔍 student_logits.shape: {student_logits.shape}")
                    logging.debug(f"🔍 teacher_logits.shape: {teacher_logits.shape}")
                    loss = kl_div_loss(student_logits, teacher_logits, vocab_size=vocab_size)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(pbar_en_zh):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                total_batches += 1
                pbar_en_zh.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            except Exception as e:
                logging.error(f"❌ 英→中训练失败 (step {step}): {e}")
                raise

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logging.info(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(output_model_dir, f"student_model_amp_shard_{shard_idx}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"💾 最佳模型已保存至: {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping 在 epoch {epoch + 1} 触发")
                break

        # 保存每轮模型
        model_path = os.path.join(output_model_dir, f"student_model_amp_shard_{shard_idx}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"💾 模型已保存至: {model_path}")

    logging.info("\n🎉 蒸馏训练完成！")
    return best_model_path

# -----------------------------
# 6. 主程序入口
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多任务蒸馏训练")
    parser.add_argument("--teacher_logits_dir", type=str, default="data/teacher_logits", help="教师logits目录")
    parser.add_argument("--output_model_dir", type=str, default="outputs/models", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--max_samples_per_task", type=int, default=None, help="每任务最大样本数")
    parser.add_argument("--device", type=str, default=None, help="计算设备")
    parser.add_argument("--use_jsonl", action="store_true", help="使用jsonl格式（默认使用.pt）")
    parser.add_argument("--teacher_model_path", type=str, default=MODEL_PATH, help="教师模型路径")
    parser.add_argument("--max_seq_len", type=int, default=64, help="最大序列长度")
    parser.add_argument("--compile", action="store_true", help="使用torch.compile加速")
    parser.add_argument("--shard_idx", type=int, default=0, help="分片索引")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping 耐心值")
    args = parser.parse_args()

    train_distill_amp(
        teacher_logits_dir=args.teacher_logits_dir,
        output_model_dir=args.output_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_samples_per_task=args.max_samples_per_task,
        device=args.device,
        use_jsonl=args.use_jsonl,
        teacher_model_path=args.teacher_model_path,
        max_seq_len=args.max_seq_len,
        compile=args.compile,
        shard_idx=args.shard_idx,
        patience=args.patience
    )