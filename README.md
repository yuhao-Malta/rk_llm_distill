# 🚀 瑞芯微端侧LLM蒸馏项目 (rk_llm_distill)

> **目标**：从 Qwen-1.5-1.8B 蒸馏出 50M~80M 学生模型，部署至 RV1126B 实现端侧翻译

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 项目概述

### 🎯 核心目标
- **教师模型**: Qwen-1.5-1.8B (1.8B 参数)
- **学生模型**: TinyTransformer (~20M 参数，可调)
- **任务**: 中英双向翻译 (zh↔en)
- **部署目标**: RV1126B 端侧芯片
- **性能指标**: BLEU>30, 延迟<30ms, 内存<512MB

### ✨ 项目特点
- ✅ **统一配置管理**: 所有参数集中在 `config/config.py`
- ✅ **规范化数据格式**: 严格遵循 `DataFormat` 定义
- ✅ **改进的 KL 散度**: 使用 padding 而非截断处理长度不匹配
- ✅ **完善的错误处理**: 日志详细，异常可追溯
- ✅ **单元测试**: 验证核心功能
- ✅ **快速启动脚本**: 一键测试完整流程

---

## 🏗️ 项目结构

```
rk_llm_distill/
├── config/
│   └── config.py              # 统一配置文件 ✨ 新增
├── models/
│   ├── tiny_transformer.py    # 学生模型 (已优化)
│   └── tokenizer_wrapper.py
├── scripts/
│   ├── generate_logits_grok.py  # 生成 teacher logits (已优化)
│   ├── evaluate_model.py        # 模型评估 (已优化)
│   ├── quantize_model.py        # 模型量化
│   └── quick_start.py           # 快速启动 ✨ 新增
├── src/
│   ├── train_distill_amp_grok.py  # 蒸馏训练 (已优化)
│   ├── coordinate_distill.py      # 协调式训练
│   ├── loss.py                    # 损失函数
│   └── evaluate.py
├── tests/
│   └── test_model.py           # 单元测试 ✨ 新增
├── data/
│   ├── raw/                    # 原始数据 (WMT19)
│   └── teacher_logits/         # Teacher logits
├── outputs/
│   ├── models/                 # 训练后的模型
│   └── logs/                   # 日志文件
├── requirements.txt            # 依赖 (已更新)
└── README.md                   # 说明文档
```

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd rk_llm_distill

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# GPU 环境 (可选)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2️⃣ 测试基础功能

```bash
# 运行测试模式 (验证环境)
python scripts/quick_start.py --mode test
```

**预期输出**:
```
✅ 单元测试通过
✅ 模型初始化成功
✅ 配置文件加载成功
```

### 3️⃣ 小规模训练 (100 样本)

```bash
# 运行小规模训练 (推荐首次使用)
python scripts/quick_start.py --mode small
```

**流程**:
1. 生成 teacher logits (100 样本)
2. 训练学生模型 (2 epochs)
3. 评估模型性能

**预计时间**: 10-20 分钟 (CPU)

### 4️⃣ 全量训练 (26M 样本)

```bash
# 全量训练 (需 GPU 服务器)
python scripts/quick_start.py --mode full
```

⚠️ **注意**: 全量训练需要:
- **GPU**: NVIDIA GPU (16GB+ 显存)
- **内存**: 32GB+ RAM
- **时间**: 数天 (取决于硬件)

---

## 📖 详细使用指南

### 手动执行各步骤

#### Step 1: 生成 Teacher Logits

```bash
# CPU 模式 (小规模测试)
python scripts/generate_logits_grok.py \
  --max_samples 1000 \
  --batch_size 2 \
  --device cpu \
  --int8

# GPU 模式 (大规模)
python scripts/generate_logits_grok.py \
  --max_samples 100000 \
  --batch_size 16 \
  --device cuda \
  --compile
```

**参数说明**:
- `--max_samples`: 最大样本数 (None=全量)
- `--batch_size`: 批大小 (CPU: 1-4, GPU: 16-32)
- `--device`: 计算设备 (cpu/cuda)
- `--int8`: 使用 INT8 量化 (CPU 推荐)
- `--compile`: 使用 torch.compile 加速 (GPU 推荐)

#### Step 2: 训练学生模型

```bash
# CPU 模式
python src/train_distill_amp_grok.py \
  --max_samples_per_task 1000 \
  --batch_size 4 \
  --epochs 3 \
  --device cpu

# GPU 模式
python src/train_distill_amp_grok.py \
  --max_samples_per_task 100000 \
  --batch_size 16 \
  --epochs 3 \
  --device cuda \
  --compile
```

**参数说明**:
- `--max_samples_per_task`: 每个任务的最大样本数
- `--epochs`: 训练轮数
- `--patience`: Early stopping 耐心值

#### Step 3: 量化模型

```bash
python scripts/quantize_model.py
```

**输出**: `outputs/models/student_model_int8.pth`

#### Step 4: 评估模型

```bash
# 评估 FP32 模型
python scripts/evaluate_model.py \
  --model_path outputs/models/student_model_amp_shard_0_best.pth \
  --max_samples 100

# 评估 INT8 模型
python scripts/evaluate_model.py \
  --model_path outputs/models/student_model_int8.pth \
  --is_int8 \
  --max_samples 100
```

**评估指标**:
- BLEU 分数 (目标: >30)
- 推理延迟 (目标: <30ms)
- 内存占用 (目标: <512MB)

---

## 🔧 配置管理

所有配置集中在 `config/config.py`：

### 模型配置

```python
# config/config.py

# 方案1: 极致压缩 (~10M 参数)
TINY_CONFIG = {
    "d_model": 96,
    "nhead": 4,
    "num_layers": 2,
    "share_weights": True
}

# 方案2: 平衡方案 (~20M 参数) - 默认
BALANCED_CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "share_weights": True
}

# 方案3: 性能优先 (~30M 参数)
PERFORMANCE_CONFIG = {
    "d_model": 192,
    "nhead": 6,
    "num_layers": 3,
    "share_weights": True
}
```

**切换配置**:
```python
# 修改 config/config.py
ModelConfig.CURRENT_CONFIG = ModelConfig.PERFORMANCE_CONFIG
```

### 训练配置

```python
class TrainingConfig:
    EPOCHS = 3
    BATCH_SIZE = 4  # CPU: 2-4, GPU: 16-32
    LEARNING_RATE = 3e-4
    TEMPERATURE = 2.0  # KL 散度温度
    PATIENCE = 2  # Early stopping
```

---

## 📊 主要改进

### 1. 统一配置管理
**问题**: 硬编码参数分散在多个文件  
**解决**: 创建 `config/config.py` 集中管理  
**优势**: 易于调整参数，避免不一致

### 2. 修复共享权重梯度问题
**问题**: `self.lm_head = self.embed` 导致梯度更新异常  
**解决**: 使用 `self.lm_head.weight = self.embed.weight`  
**优势**: 保持共享但独立模块

### 3. 规范化数据格式
**问题**: 数据键名不统一 (src_input_ids/input_ids 混用)  
**解决**: 定义 `DataFormat.REQUIRED_KEYS`，强制统一  
**优势**: 减少数据读取错误

### 4. 改进 KL 散度计算
**问题**: 序列长度不匹配时直接截断，损失信息  
**解决**: 使用 padding 补齐短序列  
**优势**: 保留完整信息，提高训练质量

### 5. 增强错误处理
**问题**: 异常难追溯  
**解决**: 详细日志 + try-except  
**优势**: 快速定位问题

### 6. 添加单元测试
**问题**: 缺少功能验证  
**解决**: `tests/test_model.py` 覆盖核心功能  
**优势**: 保证代码质量

---

## 🧪 单元测试

```bash
# 运行所有测试
python tests/test_model.py

# 使用 pytest (推荐)
pytest tests/test_model.py -v
```

**测试覆盖**:
- ✅ 模型初始化
- ✅ 前向传播
- ✅ 序列截断
- ✅ 任务嵌入
- ✅ 权重共享
- ✅ Attention mask
- ✅ 参数统计
- ✅ 批处理

---

## 📈 性能优化建议

### CPU 优化
```python
# config/config.py
TrainingConfig.BATCH_SIZE = 2
DeviceConfig.CPU_CONFIG = {
    "int8": True,  # 启用 INT8 量化
    "num_workers": 0
}
```

### GPU 优化
```python
# config/config.py
TrainingConfig.BATCH_SIZE = 16
TrainingConfig.USE_COMPILE = True  # torch.compile
DeviceConfig.GPU_CONFIG = {
    "compile": True,
    "num_workers": 4
}
```

---

## 🐛 常见问题

### Q1: 内存不足 (OOM)
**解决**:
- 减小 `batch_size`
- 减小 `max_seq_len`
- 启用 `gradient_accumulation_steps`
- 使用 INT8 量化

### Q2: BLEU 分数低 (<20)
**解决**:
- 增加训练样本数
- 增加训练轮数
- 调整学习率
- 检查数据质量

### Q3: 训练速度慢
**解决**:
- 使用 GPU
- 启用 `torch.compile`
- 增大 `batch_size`
- 减少日志输出

### Q4: 词汇表大小不匹配
**解决**:
- 检查 `config.json` 中的 `vocab_size`
- 确保 tokenizer 版本一致
- 重新生成 teacher logits

---

## 📦 部署到 RV1126B

### 1. 转换为 RKNN 格式

```bash
# 安装 RKNN Toolkit
pip install rknn-toolkit2

# 转换模型
python scripts/convert_to_rknn.py \
  --input outputs/models/student_model_int8.pth \
  --output outputs/models/student_model.rknn \
  --target rv1126
```

### 2. 板端测试

```bash
# 上传模型到板子
scp outputs/models/student_model.rknn root@<板子IP>:/root/

# SSH 登录板子
ssh root@<板子IP>

# 运行推理
./rknn_inference student_model.rknn input.txt
```

### 3. 性能验证
- **BLEU**: >30
- **延迟**: <30ms
- **内存**: <512MB

---

## 📚 参考资料

- [Qwen 模型](https://github.com/QwenLM/Qwen)
- [知识蒸馏 (Knowledge Distillation)](https://arxiv.org/abs/1503.02531)
- [WMT19 数据集](https://huggingface.co/datasets/wmt/wmt19)
- [瑞芯微 RKNN 文档](https://github.com/rockchip-linux/rknn-toolkit2)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 License

MIT License

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**祝您使用愉快！** 🎉
