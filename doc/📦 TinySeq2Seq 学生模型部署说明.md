# 📦 TinySeq2Seq 学生模型部署说明

## 🧠 模型信息

| 项目         | 内容                                          |
| ------------ | --------------------------------------------- |
| 模型名称     | TinySeq2Seq Transformer (Distilled from Qwen) |
| 框架版本     | PyTorch 2.x                                   |
| 模型精度     | FP32                                          |
| 模型文件     | `student_model.onnx`                          |
| 任务         | 中英互译 (zh↔en)                              |
| 最大序列长度 | 64                                            |
| 词表大小     | 与教师模型一致（参见 `vocab.json`）           |

---

## ⚙️ 输入输出定义

| 名称             | 形状               | 类型  | 说明                               |
| ---------------- | ------------------ | ----- | ---------------------------------- |
| `input_ids`      | `[batch, seq_len]` | int64 | tokenized 输入序列                 |
| `attention_mask` | `[batch, seq_len]` | int64 | padding 掩码                       |
| `task_id`        | `[batch]`          | int64 | 任务方向标识（0: zh→en, 1: en→zh） |

输出：
| 名称     | 形状                           | 类型    | 说明                  |
| -------- | ------------------------------ | ------- | --------------------- |
| `logits` | `[batch, seq_len, vocab_size]` | float32 | 每个 token 的概率分布 |

---

## 🧩 校准数据集

| 文件名                   | 格式                       | 说明                     |
| ------------------------ | -------------------------- | ------------------------ |
| `calibration_inputs.npy` | NumPy Array `[N, seq_len]` | Tokenized 输入样本       |
| `calibration_texts.txt`  | 文本文件                   | 对应原始输入文本（可选） |

- 样本数约 200 条
- 已经过滤异常输入，覆盖中英文任务
- 由蒸馏训练集随机采样生成

> ⚠️ 用于 RKNN Toolkit2 量化校准阶段，统计激活范围（Post-Training Quantization）。

---

## 🧰 交付文件清单

```

├── student_model.onnx
├── vocab.json
├── tokenizer_config.json
├── calibration_dataset/
│   ├── calibration_inputs.npy
│   └── calibration_texts.txt
└── README_deploy.md

```

---

## 🚀 量化部署建议

1. 使用 **RKNN Toolkit2** 导入 `student_model.onnx`

2. 指定输入节点 `input_ids`, `attention_mask`, `task_id`

3. 执行 **Post-Training Quantization (PTQ)**

   ```
   
   rknn.config(mean_values=[0], std_values=[1])
   rknn.build(do_quantization=True, dataset='calibration_dataset/calibration_inputs.npy')

4. 生成 `student_model_int8.rknn`  
在 RV1126B 平台上部署推理。

---

## 📩 联系方式

模型研发负责人: 于浩  
交付时间: （日期）  
版本: v1.0