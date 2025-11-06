import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging

from models.tiny_transformer import TinyTransformer
from config.config import (
    ModelConfig, TrainingConfig, DataFormat, LogConfig,
    MODEL_PATH, TEACHER_LOGITS_DIR, OUTPUT_MODEL_DIR, CACHE_DIR
)

# 配置日志
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
    """
    翻译数据集 (统一格式)
    直接从 .pt 文件加载数据 (字段名已规范化)
    """

    def __init__(self, file_path, max_samples=None):
        self.file_path = file_path
        self.max_samples = max_samples

        logging.info(f"📥 加载数据: {file_path}")
        data = torch.load(file_path)
        self.data = data[:max_samples] if max_samples else data

        # ✅ 验证数据格式
        if len(self.data) > 0:
            sample = self.data[0]
            for key in DataFormat.REQUIRED_KEYS:
                if key not in sample:
                    raise KeyError(f"❌ 数据缺少必需字段: {key}")

        logging.info(f"✅ 数据加载完成 (样本数: {len(self.data)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==================== collate_fn ====================
def custom_collate_fn(batch, max_seq_len=64, pad_token_id=151643):
    """批次整理 (统一字段名)"""
    # 所有字段都使用标准名称
    batch_dict = {
        "id": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "src_input_ids": torch.stack([item["src_input_ids"][:max_seq_len] for item in batch]),
        "src_attention_mask": torch.stack([item["src_attention_mask"][:max_seq_len] for item in batch]),
        "tgt_input_ids": torch.stack([item["tgt_input_ids"][:max_seq_len] for item in batch]),
        "tgt_attention_mask": torch.stack([item["tgt_attention_mask"][:max_seq_len] for item in batch]),
        "task_id": torch.tensor([item["task_id"] for item in batch], dtype=torch.long)
    }

    # 可选字段
    if "logits" in batch[0] and batch[0]["logits"] is not None:
        batch_dict["logits"] = torch.stack([item["logits"][:max_seq_len, :ModelConfig.VOCAB_SIZE] for item in batch])

    return batch_dict


# -----------------------------
# 3. KL 散度损失函数
# -----------------------------
def kl_div_loss(student_logits, teacher_logits, temperature=2.0, vocab_size=None):
    """
    知识蒸馏 KL 散度损失 (改进版)

    改进点:
    1. ✅ 使用 padding 而非截断处理长度不匹配
    2. ✅ 动态裁剪词汇表维度
    3. ✅ 增加数值稳定性检查
    """
    vocab_size = vocab_size or ModelConfig.VOCAB_SIZE
    batch_size = student_logits.size(0)
    seq_len_s = student_logits.size(1)
    seq_len_t = teacher_logits.size(1)
    vocab_s = student_logits.size(2)
    vocab_t = teacher_logits.size(2)

    # 1. 处理序列长度不匹配 (padding 而非截断)
    if seq_len_s != seq_len_t:
        max_len = max(seq_len_s, seq_len_t)
        if seq_len_s < max_len:
            pad = torch.zeros(batch_size, max_len - seq_len_s, vocab_s, device=student_logits.device)
            student_logits = torch.cat([student_logits, pad], dim=1)
        if seq_len_t < max_len:
            pad = torch.zeros(batch_size, max_len - seq_len_t, vocab_t, device=teacher_logits.device)
            teacher_logits = torch.cat([teacher_logits, pad], dim=1)
        logging.debug(f"⚠️ 序列长度 padding: {seq_len_s}, {seq_len_t} → {max_len}")

    # 2. 处理词汇表大小不匹配
    if vocab_t != vocab_size:
        teacher_logits = teacher_logits[:, :, :vocab_size]
        logging.debug(f"⚠️ Teacher 词汇表裁剪: {vocab_t} → {vocab_size}")

    # 3. 计算 KL 散度
    s_dist = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    t_dist = nn.functional.softmax(teacher_logits / temperature, dim=-1)

    # 数值稳定性检查
    if torch.isnan(s_dist).any() or torch.isnan(t_dist).any():
        logging.warning("⚠️ 检测到 NaN，跳过此 batch")
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    loss = nn.functional.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)
    return loss


# ==================== 训练主函数 ====================
def train_distill_amp(
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
    """
    多任务蒸馏训练 (AMP + 统一格式)

    改进点:
    1. ✅ 统一数据字段名
    2. ✅ 改进 KL 散度计算
    3. ✅ 增强错误处理
    4. ✅ 自动从配置文件读取参数
    """
    # 从配置文件获取默认值
    epochs = epochs or TrainingConfig.EPOCHS
    batch_size = batch_size or TrainingConfig.BATCH_SIZE
    gradient_accumulation_steps = gradient_accumulation_steps or TrainingConfig.GRADIENT_ACCUMULATION_STEPS
    learning_rate = learning_rate or TrainingConfig.LEARNING_RATE
    patience = patience or TrainingConfig.PATIENCE
    max_seq_len = max_seq_len or ModelConfig.MAX_SEQ_LEN

    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    logging.info("=" * 60)
    logging.info("🚀 开始多任务蒸馏训练")
    logging.info("=" * 60)
    logging.info(f"📦 设备: {device}")
    logging.info(f"📊 训练配置: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}")
    logging.info(f"🔧 AMP: {TrainingConfig.USE_AMP}, Compile: {compile or TrainingConfig.USE_COMPILE}")

    # 1. 初始化学生模型
    try:
        model = TinyTransformer(
            vocab_size=ModelConfig.VOCAB_SIZE,
            max_seq_len=max_seq_len,
            **ModelConfig.CURRENT_CONFIG
        ).to(device)

        if compile and device.type == "cuda":
            model = torch.compile(model)

        model.num_parameters(verbose=True)
    except Exception as e:
        logging.error(f"❌ 模型初始化失败: {e}")
        raise

    # 2. 加载数据集
    zh_to_en_path = os.path.join(teacher_logits_dir, f"zh_to_en_shard_{shard_idx}.pt")
    en_to_zh_path = os.path.join(teacher_logits_dir, f"en_to_zh_shard_{shard_idx}.pt")

    try:
        dataset_zh_en = TranslationDataset(zh_to_en_path, max_samples=max_samples_per_task)
        dataset_en_zh = TranslationDataset(en_to_zh_path, max_samples=max_samples_per_task)
    except Exception as e:
        logging.error(f"❌ 数据集加载失败: {e}")
        raise

    # 3. DataLoader
    loader_zh_en = DataLoader(
        dataset_zh_en,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: custom_collate_fn(b, max_seq_len=max_seq_len, pad_token_id=model.tokenizer.pad_token_id)
    )
    loader_en_zh = DataLoader(
        dataset_en_zh,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: custom_collate_fn(b, max_seq_len=max_seq_len, pad_token_id=model.tokenizer.pad_token_id)
    )

    # 4. 优化器 + AMP
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and TrainingConfig.USE_AMP))

    # 5. 训练循环
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"📅 Epoch {epoch + 1}/{epochs}")
        logging.info(f"{'=' * 60}")

        total_loss = 0.0
        total_batches = 0

        # 中→英训练
        logging.info("🧠 训练中→英任务...")
        pbar = tqdm(loader_zh_en, desc="中→英")
        for step, batch in enumerate(pbar):
            try:
                # 数据迁移到设备
                src_input_ids = batch["src_input_ids"].to(device)
                src_attention_mask = batch["src_attention_mask"].to(device)
                task_ids = batch["task_id"].to(device)
                teacher_logits = batch["logits"].to(device) if "logits" in batch else None

                # 前向传播
                with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=TrainingConfig.USE_AMP):
                    outputs = model(
                        input_ids=src_input_ids,
                        attention_mask=src_attention_mask,
                        task_id=task_ids
                    )
                    student_logits = outputs["logits"]

                    # 计算损失
                    if teacher_logits is not None:
                        loss = kl_div_loss(student_logits, teacher_logits,
                                           temperature=TrainingConfig.TEMPERATURE,
                                           vocab_size=ModelConfig.VOCAB_SIZE)
                    else:
                        logging.warning("⚠️ 无 teacher_logits，跳过")
                        continue

                    loss = loss / gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                total_batches += 1
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

            except Exception as e:
                logging.error(f"❌ 中→英训练失败 (step {step}): {e}")
                continue

        # 英→中训练
        logging.info("🧠 训练英→中任务...")
        pbar = tqdm(loader_en_zh, desc="英→中")
        for step, batch in enumerate(pbar):
            try:
                src_input_ids = batch["src_input_ids"].to(device)
                src_attention_mask = batch["src_attention_mask"].to(device)
                task_ids = batch["task_id"].to(device)
                teacher_logits = batch["logits"].to(device) if "logits" in batch else None

                with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=TrainingConfig.USE_AMP):
                    outputs = model(
                        input_ids=src_input_ids,
                        attention_mask=src_attention_mask,
                        task_id=task_ids
                    )
                    student_logits = outputs["logits"]

                    if teacher_logits is not None:
                        loss = kl_div_loss(student_logits, teacher_logits,
                                           temperature=TrainingConfig.TEMPERATURE,
                                           vocab_size=ModelConfig.VOCAB_SIZE)
                    else:
                        continue

                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                total_batches += 1
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

            except Exception as e:
                logging.error(f"❌ 英→中训练失败 (step {step}): {e}")
                continue

        # 计算平均损失
        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        logging.info(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(output_model_dir, f"student_model_amp_shard_{shard_idx}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"💾 最佳模型已保存: {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping 触发 (Epoch {epoch + 1})")
                break

        # 保存每轮模型
        model_path = os.path.join(output_model_dir, f"student_model_amp_shard_{shard_idx}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"💾 模型已保存: {model_path}")

        # ✅ 仅保留最佳模型，立即删除当前epoch文件
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(output_model_dir, f"student_model_amp_shard_{shard_idx}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🏆 新最佳模型已保存: {best_model_path}")
        else:
            # 删除当前epoch文件（非最佳）
            os.remove(model_path)
            logging.info(f"🗑️ 删除非最佳模型: {model_path}")

    logging.info("\n🎉 训练完成！")
    return best_model_path if best_loss < float('inf') else model_path


# ==================== 主程序 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多任务蒸馏训练 (优化版)")
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

    train_distill_amp(
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