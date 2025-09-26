import torch
import torch.nn.functional as F


def kl_div_loss(student_logits, teacher_logits, temperature=2.0):
    """
    知识蒸馏 KL 散度损失
    """
    s_dist = F.log_softmax(student_logits / temperature, dim=-1)
    t_dist = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_dist, t_dist, reduction='batchmean') * (temperature ** 2)


if __name__ == "__main__":
    # 仿真数据
    student_logits = torch.randn(8, 32, 32000)
    teacher_logits = torch.randn(8, 32, 32000)

    loss = kl_div_loss(student_logits, teacher_logits)
    print(f"✅ KL Loss: {loss.item():.4f}")