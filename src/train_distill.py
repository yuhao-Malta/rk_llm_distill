# src/train_distill.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.tiny_transformer import TinyTransformer  # 👈 你的学生模型


# -----------------------------
# 1. 自定义数据集类
# -----------------------------
class TranslationDataset(Dataset):
    """加载教师软标签数据（JSONL 格式）"""

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
            "hyp": item["hyp"],
            "task_id": item["task_id"],  # 0=中→英, 1=英→中
            "timestamp": item["timestamp"]
        }


# -----------------------------
# 2. KL 散度损失函数
# -----------------------------
def kl_div_loss(student_logits, teacher_logits, temperature=2.0):
    """
    知识蒸馏 KL 散度损失
    :param student_logits: 学生模型输出 (batch_size, seq_len, vocab_size)
    :param teacher_logits: 教师模型输出 (batch_size, seq_len, vocab_size)
    :param temperature: 温度系数，控制软标签平滑度
    :return: KL 散度损失
    """
    s_dist = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    t_dist = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return nn.functional.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)


# -----------------------------
# 3. 蒸馏训练主函数
# -----------------------------
def train_distill(
        teacher_logits_dir="../scripts/data/teacher_logits",
        output_model_dir="../outputs/models",
        epochs=3,
        batch_size=8,
        learning_rate=3e-4,
        max_samples_per_task=None  # 限制每个任务的样本数（调试用）
):
    """
    多任务蒸馏训练（中→英 + 英→中）
    """
    print("🚀 开始多任务蒸馏训练...")

    # 1. 创建输出目录
    os.makedirs(output_model_dir, exist_ok=True)

    # 2. 初始化学生模型
    model = TinyTransformer()
    tokenizer = model.tokenizer  # 复用 Qwen Tokenizer

    # 3. 加载数据集
    zh_to_en_path = os.path.join(teacher_logits_dir, "zh_to_en.jsonl")
    en_to_zh_path = os.path.join(teacher_logits_dir, "en_to_zh.jsonl")

    print(f"🔍 当前工作目录: {os.getcwd()}")
    print(f"🔍 尝试加载中→英数据: {os.path.abspath(zh_to_en_path)}")
    print(f"🔍 尝试加载英→中数据: {os.path.abspath(en_to_zh_path)}")

    if not os.path.exists(zh_to_en_path):
        raise FileNotFoundError(f"❌ 未找到中→英数据: {zh_to_en_path}")
    if not os.path.exists(en_to_zh_path):
        raise FileNotFoundError(f"❌ 未找到英→中数据: {en_to_zh_path}")

    dataset_zh_en = TranslationDataset(zh_to_en_path, max_samples=max_samples_per_task)
    dataset_en_zh = TranslationDataset(en_to_zh_path, max_samples=max_samples_per_task)

    loader_zh_en = DataLoader(dataset_zh_en, batch_size=batch_size, shuffle=True)
    loader_en_zh = DataLoader(dataset_en_zh, batch_size=batch_size, shuffle=True)

    print(f"📊 数据集统计:")
    print(f"  - 中→英样本数: {len(dataset_zh_en)}")
    print(f"  - 英→中样本数: {len(dataset_en_zh)}")

    # 4. 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    # 5. 训练循环
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        total_batches = 0

        # --- 中→英训练 ---
        print("🧠 训练中→英任务...")
        pbar_zh_en = tqdm(loader_zh_en, desc="中→英", leave=False)
        for batch in pbar_zh_en:
            optimizer.zero_grad()

            # 使用模型的 tokenizer 编码输入
            encoded = model.tokenizer(
                batch["src"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.max_seq_len  # 👈 保持一致
            )
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            # ✅ 调试：打印序列长度
            seq_len = input_ids.size(1)
            if seq_len > 64:
                print(f"🚨 警告：发现超长序列！长度={seq_len}，内容前50字符: {batch['src'][0][:50]}...")

            # 前向传播（学生模型）
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            task_id=torch.tensor([0] * input_ids.size(0)))  # task_id=0
            student_logits = outputs["logits"]

            # 模拟教师 logits（实际应从文件加载）
            # TODO: 从 batch["hyp"] 生成真实 teacher_logits（需 tokenizer 编码）
            # 这里仅为示意，实际应加载真实 teacher_logits
            teacher_logits = torch.randn_like(student_logits)  # 👈 占位符

            # 计算损失
            loss = kl_div_loss(student_logits, teacher_logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar_zh_en.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 英→中训练 ---
        print("🧠 训练英→中任务...")
        pbar_en_zh = tqdm(loader_en_zh, desc="英→中", leave=False)
        for batch in pbar_en_zh:
            optimizer.zero_grad()

            # 编码输入
            encoded = tokenizer(batch["src"], return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            # ✅ 调试：打印序列长度
            seq_len = input_ids.size(1)
            if seq_len > 64:
                print(f"🚨 警告：发现超长序列！长度={seq_len}，内容前50字符: {batch['src'][0][:50]}...")

            # 前向传播（学生模型）
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            task_id=torch.tensor([1] * input_ids.size(0)))  # task_id=1
            student_logits = outputs["logits"]

            # 模拟教师 logits（实际应从文件加载）
            teacher_logits = torch.randn_like(student_logits)  # 👈 占位符

            # 计算损失
            loss = kl_div_loss(student_logits, teacher_logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar_en_zh.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

        # 6. 保存模型（每个 epoch）
        model_path = os.path.join(output_model_dir, f"student_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已保存至: {model_path}")

    print("\n🎉 蒸馏训练完成！")


# -----------------------------
# 4. 主程序入口
# -----------------------------
if __name__ == "__main__":
    train_distill(
        teacher_logits_dir="../scripts/data/teacher_logits",
        output_model_dir="../outputs/models",
        epochs=3,  # 👈 可调整
        batch_size=8,  # 👈 根据 GPU 内存调整
        learning_rate=3e-4,
        max_samples_per_task=100  # 👈 调试时限制样本数
    )