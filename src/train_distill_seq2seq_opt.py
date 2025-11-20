import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tiny_seq2seq_transformer import TinySeq2SeqTransformer
from config.config import (
    ModelConfig, TrainingConfig, DataFormat, LogConfig,
    OPUS_MT_ZH_EN, OPUS_MT_EN_ZH,
    TEACHER_LOGITS_DIR, OUTPUT_MODEL_DIR
)

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format=LogConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LogConfig.TRAIN_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==================== 数据集类 ====================
class TranslationDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.file_path = file_path
        data = torch.load(file_path)
        self.data = data[:max_samples] if max_samples else data

        if len(self.data) > 0:
            for key in DataFormat.REQUIRED_KEYS:
                if key not in self.data[0]:
                    raise KeyError(f"❌ 缺少字段: {key}")

        logging.info(f"✅ 加载数据: {file_path} ({len(self.data)} samples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==================== collate ====================
def custom_collate_fn(batch, max_seq_len=64, pad_token_id=0):
    batch_dict = {
        "id": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "src_input_ids": torch.stack([item["src_input_ids"][:max_seq_len] for item in batch]),
        "src_attention_mask": torch.stack([item["src_attention_mask"][:max_seq_len] for item in batch]),
        "tgt_input_ids": torch.stack([item["tgt_input_ids"][:max_seq_len] for item in batch]),
        "tgt_attention_mask": torch.stack([item["tgt_attention_mask"][:max_seq_len] for item in batch]),
        "task_id": torch.tensor([item["task_id"] for item in batch], dtype=torch.long)
    }

    if "logits" in batch[0] and batch[0]["logits"] is not None:
        batch_dict["logits"] = torch.stack([
            item["logits"][:max_seq_len, :ModelConfig.VOCAB_SIZE] for item in batch
        ])

    return batch_dict

# ==================== KL Loss ====================
def kl_div_loss(student_logits, teacher_logits, temperature=2.0):
    s_dist = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    t_dist = nn.functional.softmax(teacher_logits / temperature, dim=-1)

    if torch.isnan(s_dist).any() or torch.isnan(t_dist).any():
        logging.warning("⚠️ 检测到 NaN，跳过 batch")
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    loss = nn.functional.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)
    return loss

# ==================== 主训练 ====================
def train_distill_seq2seq(
    teacher_logits_dir=TEACHER_LOGITS_DIR,
    output_model_dir=OUTPUT_MODEL_DIR,
    epochs=None,
    batch_size=None,
    gradient_accumulation_steps=None,
    learning_rate=None,
    max_samples_per_task=None,
    device=None,
    max_seq_len=None,
    compile=None,
    shard_idx=0,
    patience=None
):
    epochs = epochs or TrainingConfig.EPOCHS
    batch_size = batch_size or TrainingConfig.BATCH_SIZE
    gradient_accumulation_steps = gradient_accumulation_steps or TrainingConfig.GRADIENT_ACCUMULATION_STEPS
    learning_rate = learning_rate or TrainingConfig.LEARNING_RATE
    patience = patience or TrainingConfig.PATIENCE
    max_seq_len = max_seq_len or ModelConfig.MAX_SEQ_LEN

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logging.info("=" * 60)
    logging.info("🚀 启动 Seq2Seq 蒸馏训练 (Opus-MT 教师)")
    logging.info("=" * 60)

    # ==== 分别加载两个方向的 tokenizer ====
    tokenizer_zh_en = AutoTokenizer.from_pretrained(OPUS_MT_ZH_EN, local_files_only=True)
    tokenizer_en_zh = AutoTokenizer.from_pretrained(OPUS_MT_EN_ZH, local_files_only=True)
    pad_id_zh_en = tokenizer_zh_en.pad_token_id or 0
    pad_id_en_zh = tokenizer_en_zh.pad_token_id or 0

    vocab_size_zh_en = len(tokenizer_zh_en)
    vocab_size_en_zh = len(tokenizer_en_zh)

    if vocab_size_zh_en != vocab_size_en_zh:
        logging.warning(f"⚠️ 双向 vocab_size 不一致: zh→en={vocab_size_zh_en}, en→zh={vocab_size_en_zh}，取最大值")
    vocab_size = max(vocab_size_zh_en, vocab_size_en_zh)

    logging.info(f"🧩 使用教师词表大小: {vocab_size}")

    logging.info("🧠 正在初始化学生模型 (TinySeq2SeqTransformer)...")

    try:
        model = TinySeq2SeqTransformer(
            teacher_model_path_zh2en=OPUS_MT_ZH_EN,
            teacher_model_path_en2zh=OPUS_MT_EN_ZH,
            d_model=ModelConfig.CURRENT_CONFIG.get("d_model", 128),
            nhead=ModelConfig.CURRENT_CONFIG.get("nhead", 4),
            num_encoder_layers=ModelConfig.CURRENT_CONFIG.get("num_encoder_layers", 2),
            num_decoder_layers=ModelConfig.CURRENT_CONFIG.get("num_decoder_layers", 2),
            dim_feedforward=ModelConfig.CURRENT_CONFIG.get("dim_feedforward", 256),
            dropout=ModelConfig.CURRENT_CONFIG.get("dropout", 0.1),
            max_seq_len=args.max_seq_len,
            share_weights=True,  # ✅ 保留权重共享
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        # logging.info(f"✅ 学生模型参数总数: {total_params:,}")
        logging.info(f"✅ TinySeq2SeqTransformer 初始化成功，学生模型参数总数:  {total_params / 1e6:.2f}M")

    except Exception as e:
        logging.error(f"❌ 学生模型加载失败: {e}")
        raise
    # ===== Sanity check =====
    if hasattr(model, "zh2en_tokenizer") and model.zh2en_tokenizer:
        logging.info(f"教师 zh→en 词表: {len(model.zh2en_tokenizer)}")
    if hasattr(model, "en2zh_tokenizer") and model.en2zh_tokenizer:
        logging.info(f"教师 en→zh 词表: {len(model.en2zh_tokenizer)}")
    logging.info(f"学生模型词表: {model.vocab_size}")

    if compile and device.type == "cuda":
        model = torch.compile(model)

    # ==== 数据路径 ====
    zh_to_en_path = os.path.join(teacher_logits_dir, f"zh_to_en_shard_{shard_idx}.pt")
    en_to_zh_path = os.path.join(teacher_logits_dir, f"en_to_zh_shard_{shard_idx}.pt")

    loader_zh_en = DataLoader(
        TranslationDataset(zh_to_en_path, max_samples=max_samples_per_task),
        batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: custom_collate_fn(b, max_seq_len=max_seq_len, pad_token_id=pad_id_zh_en)
    )
    loader_en_zh = DataLoader(
        TranslationDataset(en_to_zh_path, max_samples=max_samples_per_task),
        batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: custom_collate_fn(b, max_seq_len=max_seq_len, pad_token_id=pad_id_en_zh)
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda" and TrainingConfig.USE_AMP))

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_zh_en)  # ✅ 使用 Opus pad_token_id

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        logging.info(f"\n{'=' * 60}\n📅 Epoch {epoch + 1}/{epochs}\n{'=' * 60}")
        total_loss = 0.0
        total_batches = 0

        for task_name, loader in [("中→英", loader_zh_en), ("英→中", loader_en_zh)]:
            logging.info(f"🧠 训练任务: {task_name}")
            pbar = tqdm(loader, desc=task_name)

            for step, batch in enumerate(pbar):
                try:
                    src_input_ids = batch["src_input_ids"].to(device)
                    src_attention_mask = batch["src_attention_mask"].to(device)
                    tgt_input_ids = batch["tgt_input_ids"].to(device)
                    tgt_attention_mask = batch["tgt_attention_mask"].to(device)
                    task_ids = batch["task_id"].to(device)

                    teacher_logits = batch.get("logits", None)
                    if teacher_logits is not None:
                        teacher_logits = teacher_logits.to(device)

                    with autocast(device_type=device.type, enabled=TrainingConfig.USE_AMP):
                        outputs = model(
                            src_ids=src_input_ids,
                            tgt_ids=tgt_input_ids,
                            task_id=task_ids,
                            src_padding_mask=(1 - src_attention_mask).bool(),
                            tgt_padding_mask=(1 - tgt_attention_mask).bool()
                        )

                        student_logits = outputs["logits"]

                        # -------------------------
                        # CE 部分（teacher forcing）
                        # -------------------------
                        ce_student_logits = student_logits[:, :-1].contiguous()
                        ce_labels = tgt_input_ids[:, 1:].contiguous()

                        ce_loss = ce_loss_fn(
                            ce_student_logits.reshape(-1, ce_student_logits.size(-1)),
                            ce_labels.reshape(-1)
                        )

                        # -------------------------
                        # KL 部分（蒸馏）
                        # -------------------------
                        if teacher_logits is not None:
                            # 对齐长度
                            min_len = min(student_logits.size(1), teacher_logits.size(1))
                            kl_loss = kl_div_loss(
                                student_logits[:, :min_len, :],
                                teacher_logits[:, :min_len, :],
                                temperature=TrainingConfig.TEMPERATURE
                            )
                        else:
                            kl_loss = torch.tensor(0.0, device=device)

                        # -------------------------
                        # 方案 B: 0.5 CE + 0.5 KL
                        # -------------------------
                        loss = 0.5 * ce_loss + 0.5 * kl_loss
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    # ========== 梯度累积 ==========
                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    total_loss += loss.item() * gradient_accumulation_steps
                    total_batches += 1
                    pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

                except Exception as e:
                    logging.error(f"❌ {task_name} step {step} 失败: {e}")
                    continue

        avg_loss = total_loss / max(total_batches, 1)
        logging.info(f"✅ Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 保存 checkpoint
        model_path = os.path.join(output_model_dir, f"student_seq2seq_shard_{shard_idx}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)

        # 判断最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(output_model_dir, f"student_seq2seq_shard_{shard_idx}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🏆 更新最佳模型: {best_model_path}")
        else:
            epochs_no_improve += 1
            os.remove(model_path)
            logging.info(f"🗑️ 删除非最佳模型: {model_path}")

            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early Stopping (Epoch {epoch + 1})")
                break

    logging.info("\n🎉 训练完成！")
    return best_model_path


# ================================
# CLI
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TinySeq2Seq 蒸馏训练")
    parser.add_argument("--teacher_logits_dir", type=str, default=TEACHER_LOGITS_DIR)
    parser.add_argument("--output_model_dir", type=str, default=OUTPUT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_samples_per_task", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    train_distill_seq2seq(
        teacher_logits_dir=args.teacher_logits_dir,
        output_model_dir=args.output_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_samples_per_task=args.max_samples_per_task,
        device=args.device,
        max_seq_len=args.max_seq_len,
        compile=args.compile,
        shard_idx=args.shard_idx,
        patience=args.patience
    )