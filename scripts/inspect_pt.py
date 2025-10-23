import torch

# 检查 .pt 文件内容
def inspect_pt_file(file_path):
    data = torch.load(file_path)
    print(f"文件: {file_path}")
    print(f"样本数: {len(data)}")
    print(f"首个样本键: {list(data[0].keys())}")
    if "logits" in data[0]:
        print(f"logits 形状: {data[0]['logits'].shape}")
    print("-" * 50)

# 检查两个 .pt 文件
inspect_pt_file("data/teacher_logits/zh_to_en_shard_0.pt")
inspect_pt_file("data/teacher_logits/en_to_zh_shard_0.pt")