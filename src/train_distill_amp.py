# src/train_distill_amp.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast  # 👈 AMP 核心模块
from tqdm import tqdm
import time

# 假设 TinyTransformer 已定义在 models/tiny_transformer.py
from models.tiny_transformer import TinyTransformer


# -----------------------------
# 1. 自定义数据集类（加载教师软标签）
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "id": item["id"],
            "src": item["src"],
            "ref": item["ref"],
            "hyp": item["hyp"],  # 教师模型的软标签（文本）
            "task_id": item["task_id"],  # 0=中→英, 1=英→中
            "timestamp": item["timestamp"]
        }


# -----------------------------
# 2. KL 散度损失函数（用于知识蒸馏）
# -----------------------------
def kl_div_loss(student_logits, teacher_logits, temperature=2.0):
    """
    知识蒸馏 KL 散度损失
    :param student_logits: 学生模型输出 (batch_size, seq_len, vocab_size)
    :param teacher_logits: 教师模型输出 (batch_size, seq_len, vocab_size)
    :param temperature: 温度系数，控制软标签平滑度
    :return: KL 散度损失
    """

    seq_len_s = student_logits.size(1)
    seq_len_t = teacher_logits.size(1)

    # ✅ 同步长度：截断到较短的序列长度
    min_len = min(seq_len_s, seq_len_t)
    if seq_len_s != seq_len_t:
        print(f"⚠️  序列长度不匹配 ({seq_len_s} vs {seq_len_t})，自动同步到 {min_len}")
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]

    # 计算 KL 散度
    s_dist = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    t_dist = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return nn.functional.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)


# -----------------------------
# 3. 蒸馏训练主函数（支持 AMP + 大 Batch + 梯度累积）
# -----------------------------
def train_distill_amp(
        teacher_logits_dir="data/teacher_logits",
        output_model_dir="outputs/models",
        epochs=3,
        batch_size=8,  # 👈 可调整（高端PC可设为64/128）
        gradient_accumulation_steps=4,  # 👈 梯度累积步数（模拟大 Batch）
        learning_rate=3e-4,
        max_samples_per_task=None,
        device=None  # 👈 自动检测设备
):
    """
    多任务蒸馏训练（中→英 + 英→中），支持 AMP + 大 Batch + 梯度累积
    """
    print("🚀 开始多任务蒸馏训练 (AMP + 大 Batch)...")

    # 1. 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 使用设备: {device}")

    # 2. 创建输出目录
    os.makedirs(output_model_dir, exist_ok=True)

    # 3. 初始化学生模型
    model = TinyTransformer().to(device)
    tokenizer = model.tokenizer  # 复用 Qwen Tokenizer

    # 4. 加载数据集
    zh_to_en_path = os.path.join(teacher_logits_dir, "zh_to_en.jsonl")
    en_to_zh_path = os.path.join(teacher_logits_dir, "en_to_zh.jsonl")

    if not os.path.exists(zh_to_en_path):
        raise FileNotFoundError(f"❌ 未找到中→英数据: {zh_to_en_path}")
    if not os.path.exists(en_to_zh_path):
        raise FileNotFoundError(f"❌ 未找到英→中数据: {en_to_zh_path}")

    dataset_zh_en = TranslationDataset(zh_to_en_path, max_samples=max_samples_per_task)
    dataset_en_zh = TranslationDataset(en_to_zh_path, max_samples=max_samples_per_task)

    loader_zh_en = DataLoader(dataset_zh_en, batch_size=batch_size, shuffle=True, num_workers=2)  # 👈 多线程加速
    loader_en_zh = DataLoader(dataset_en_zh, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"📊 数据集统计:")
    print(f"  - 中→英样本数: {len(dataset_zh_en)}")
    print(f"  - 英→中样本数: {len(dataset_en_zh)}")

    # 5. 初始化优化器、AMP 缩放器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # 👈 AMP 核心：梯度缩放器

    # 6. 训练循环
    model.train()
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        total_batches = 0

        # --- 中→英训练 ---
        print("🧠 训练中→英任务...")
        pbar_zh_en = tqdm(loader_zh_en, desc="中→英", leave=False)
        for step, batch in enumerate(pbar_zh_en):
            # 编码输入
            encoded = tokenizer(batch["src"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            task_ids = torch.tensor([0] * input_ids.size(0), device=device)  # task_id=0 for zh→en

            # 模拟教师 logits（实际应从 batch["hyp"] 生成，此处为示意）
            # TODO: 从 batch["hyp"] 编码生成真实 teacher_logits
            teacher_logits = torch.randn(input_ids.size(0), input_ids.size(1), model.vocab_size).to(device)

            # ✅ AMP 上下文管理器
            with autocast():  # 👈 自动混合精度
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_id=task_ids)
                student_logits = outputs["logits"]
                # ... 在计算 loss 前 ...
                print(f"🔍 student_logits.shape: {student_logits.shape}")
                print(f"🔍 teacher_logits.shape: {teacher_logits.shape}")
                loss = kl_div_loss(student_logits, teacher_logits)
                loss = loss / gradient_accumulation_steps  # 👈 梯度累积

            # 反向传播（AMP 缩放）
            scaler.scale(loss).backward()

            # 梯度累积步数到达后，更新权重
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(pbar_zh_en):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            total_batches += 1
            pbar_zh_en.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        # --- 英→中训练 ---
        print("🧠 训练英→中任务...")
        pbar_en_zh = tqdm(loader_en_zh, desc="英→中", leave=False)
        for step, batch in enumerate(pbar_en_zh):
            # 编码输入
            encoded = tokenizer(batch["src"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            task_ids = torch.tensor([1] * input_ids.size(0), device=device)  # task_id=1 for en→zh

            # 模拟教师 logits
            teacher_logits = torch.randn(input_ids.size(0), input_ids.size(1), model.vocab_size).to(device)

            # ✅ AMP 上下文管理器
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_id=task_ids)
                student_logits = outputs["logits"]
                # ... 在计算 loss 前 ...
                print(f"🔍 student_logits.shape: {student_logits.shape}")
                print(f"🔍 teacher_logits.shape: {teacher_logits.shape}")
                loss = kl_div_loss(student_logits, teacher_logits)
                loss = loss / gradient_accumulation_steps

            # 反向传播（AMP 缩放）
            scaler.scale(loss).backward()

            # 梯度累积步数到达后，更新权重
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(pbar_en_zh):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            total_batches += 1
            pbar_en_zh.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

        # 7. 保存模型（每个 epoch）
        model_path = os.path.join(output_model_dir, f"student_model_amp_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已保存至: {model_path}")

    print("\n🎉 蒸馏训练完成！")


# -----------------------------
# 4. 主程序入口
# -----------------------------
if __name__ == "__main__":
    train_distill_amp(
        teacher_logits_dir="../scripts/data/teacher_logits",
        output_model_dir="../outputs/models",
        epochs=3,
        batch_size=8,  # 👈 普通PC建议8，高端PC可设为64/128
        gradient_accumulation_steps=4,  # 👈 梯度累积步数
        learning_rate=3e-4,
        max_samples_per_task=100  # 👈 调试用，可设为 None
    )