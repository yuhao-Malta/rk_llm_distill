# config/config.py
"""
项目统一配置文件
集中管理所有超参数和路径配置
"""
import os
import json

# ==================== 路径配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")
OPUS_MT_ZH_EN = os.path.join(PROJECT_ROOT, "models", "opus_mt_zh_en")
OPUS_MT_EN_ZH = os.path.join(PROJECT_ROOT, "models", "opus_mt_en_zh")

# 数据路径
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_ROOT, "raw", "wmt19_zh_en")
TEACHER_LOGITS_DIR = os.path.join(DATA_ROOT, "teacher_logits")
CACHE_DIR = os.path.join(TEACHER_LOGITS_DIR, "cache")

# 输出路径
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
OUTPUT_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# 创建必要目录
for path in [TEACHER_LOGITS_DIR, CACHE_DIR, OUTPUT_MODEL_DIR, OUTPUT_LOGS_DIR]:
    os.makedirs(path, exist_ok=True)

# ==================== 模型配置 ====================
class ModelConfig:
    """学生模型配置"""
    # 从 Qwen config.json 动态读取词汇表大小
    @staticmethod
    def get_vocab_size():
        config_path = os.path.join(MODEL_PATH, "config.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("vocab_size", 151936)
        except Exception as e:
            print(f"⚠️ 读取 config.json 失败: {e}，使用默认值 151936")
            return 151936
    
    VOCAB_SIZE = get_vocab_size.__func__()  # 静态调用
    MAX_SEQ_LEN = 64

    # TinySeq2SeqTransformer 参数配置 (适配 encoder-decoder 架构)

    # 方案1: 极致压缩 (~10M 参数)
    TINY_CONFIG = {
        "d_model": 96,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 192,
        "dropout": 0.1,
        "share_weights": True
    }

    # 方案2: 平衡方案 (~20M 参数) - 默认
    BALANCED_CONFIG = {
        "d_model": 128,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "share_weights": True
    }

    # 方案3: 性能优先 (~30M 参数)
    PERFORMANCE_CONFIG = {
        "d_model": 192,
        "nhead": 6,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "share_weights": True
    }

    # 当前使用的配置
    CURRENT_CONFIG = BALANCED_CONFIG

# ==================== 训练配置 ====================
class TrainingConfig:
    """训练超参数"""
    EPOCHS = 3
    BATCH_SIZE = 4  # CPU 默认值，GPU 建议 16-32
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 3e-4
    TEMPERATURE = 2.0  # KL 散度温度
    PATIENCE = 2  # Early stopping
    
    # AMP 配置
    USE_AMP = True
    USE_COMPILE = False  # torch.compile (需 PyTorch 2.0+)
    
    # 数据配置
    MAX_SAMPLES_PER_TASK = None  # None = 全量
    SHARD_SIZE = 100000  # 分片大小

# ==================== 数据格式规范 ====================
class DataFormat:
    """统一数据格式定义"""
    # 标准键名 (强制使用)
    REQUIRED_KEYS = [
        "id",                    # 样本ID
        "src_text",              # 源语言文本
        "tgt_text",              # 目标语言文本 (参考翻译)
        "src_input_ids",         # 源语言token IDs
        "src_attention_mask",    # 源语言attention mask
        "tgt_input_ids",         # 目标语言token IDs
        "tgt_attention_mask",    # 目标语言attention mask
        "task_id",               # 任务ID (0=中→英, 1=英→中)
    ]
    
    # 可选键名
    OPTIONAL_KEYS = [
        "logits",                # Teacher logits (本地模式)
        "hyp_text",              # 翻译文本 (API模式)
        "timestamp",             # 生成时间戳
    ]
    
    # 任务映射
    TASK_MAP = {
        "zh_to_en": 0,
        "en_to_zh": 1
    }

# ==================== 设备配置 ====================
class DeviceConfig:
    """计算设备配置"""
    import torch
    
    # 自动检测
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    
    # CPU 优化配置
    CPU_CONFIG = {
        "batch_size": 2,
        "num_workers": 0,
        "int8": True,
        "compile": False
    }
    
    # GPU 优化配置
    GPU_CONFIG = {
        "batch_size": 16,
        "num_workers": 4,
        "int8": False,
        "compile": True
    }

# ==================== 日志配置 ====================
class LogConfig:
    """日志配置"""
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 日志文件路径
    GENERATE_LOG = os.path.join(OUTPUT_LOGS_DIR, "generate_logits.log")
    TRAIN_LOG = os.path.join(OUTPUT_LOGS_DIR, "train_distill_amp.log")
    EVALUATE_LOG = os.path.join(OUTPUT_LOGS_DIR, "evaluate_model.log")
    COORDINATE_LOG = os.path.join(OUTPUT_LOGS_DIR, "coordinate_distill.log")

# ==================== 评估配置 ====================
class EvalConfig:
    """评估配置"""
    TEST_DATA_PATH = os.path.join(RAW_DATA_PATH, "validation", "validation-00000-of-00001.parquet")
    MAX_EVAL_SAMPLES = 100
    TASKS = ["zh_to_en", "en_to_zh"]
    
    # 性能目标 (RV1126B)
    TARGET_BLEU = 30.0
    TARGET_LATENCY_MS = 30.0
    TARGET_MEMORY_MB = 512.0

# ==================== 辅助函数 ====================
def get_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("📋 项目配置摘要")
    print("=" * 60)
    print(f"📦 词汇表大小: {ModelConfig.VOCAB_SIZE}")
    print(f"📏 最大序列长度: {ModelConfig.MAX_SEQ_LEN}")
    print(f"🧠 模型配置: {ModelConfig.CURRENT_CONFIG}")
    print(f"🔧 训练配置: Epochs={TrainingConfig.EPOCHS}, Batch={TrainingConfig.BATCH_SIZE}")
    print(f"💻 设备: {DeviceConfig.DEFAULT_DEVICE}")
    print(f"📂 数据路径: {RAW_DATA_PATH}")
    print(f"💾 输出路径: {OUTPUT_MODEL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    get_config_summary()
