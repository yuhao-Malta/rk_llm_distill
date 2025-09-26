import os
import logging
from tqdm import tqdm
from datasets import load_dataset, DownloadConfig

# ✅ 强制设置环境变量（在导入 datasets 前设置！）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ✅ 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ 创建 logs 和 data/raw 目录
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'download.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def download_wmt_data(save_path=DATA_DIR, max_samples=None):
    """
    从 HF-Mirror 下载 WMT19 中英平行语料
    """
    try:
        logging.info("🌐 开始从 HF-Mirror 下载 WMT19 中英数据...")

        # ✅ 创建 DownloadConfig 对象（移除 download_from_mirror）
        download_config = DownloadConfig(
            use_etag=False  # 👈 只保留有效参数
        )

        # ✅ 使用 DownloadConfig 对象
        dataset = load_dataset(
            "wmt/wmt19",
            "zh-en",
            download_config=download_config
        )

        # 如果指定最大样本数，进行截断
        if max_samples:
            for split in dataset.keys():
                dataset[split] = dataset[split].select(range(min(max_samples, len(dataset[split]))))
            logging.info(f"✂️  截断至 {max_samples} 条样本")

        # 保存数据集
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)

        # 打印统计信息
        stats = {split: len(dataset[split]) for split in dataset.keys()}
        logging.info(f"📊 数据集统计: {stats}")
        logging.info(f"✅ 数据集保存至: {save_path}")

        return dataset

    except Exception as e:
        logging.error(f"❌ 下载失败: {str(e)}")
        raise


def validate_data(save_path=DATA_DIR):
    """
    验证数据完整性
    """
    try:
        from datasets import load_from_disk
        sample_path = os.path.join(save_path, "train")
        dataset = load_from_disk(sample_path)

        # 检查关键字段
        sample = dataset[0]
        assert 'translation' in sample, "缺少 translation 字段"
        assert 'zh' in sample['translation'], "缺少中文字段"
        assert 'en' in sample['translation'], "缺少英文字段"

        logging.info("✅ 数据验证通过！")
        return True

    except Exception as e:
        logging.error(f"❌ 数据验证失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 下载数据（先下载1000条用于测试）
    dataset = download_wmt_data(max_samples=1000)

    # 验证数据
    validate_data()