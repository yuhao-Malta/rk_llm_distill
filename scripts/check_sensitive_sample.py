# scripts/check_sensitive_sample.py

import argparse
from datasets import Dataset
import glob
import os
import sys


def load_dataset(parquet_file_path=None):
    """加载数据集"""
    try:
        if parquet_file_path:
            # 如果提供了具体的文件路径，则加载该文件
            print(f"Loading dataset from provided file: {parquet_file_path}")
            dataset = Dataset.from_parquet(parquet_file_path)
        else:
            # 否则，尝试从默认目录加载第一个文件
            default_data_path = "data/raw/wmt19_zh_en/train/"
            if not os.path.exists(default_data_path):
                # 如果默认路径不存在，尝试基于项目根目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_data_path = os.path.join(project_root, default_data_path)

            print(f"Searching for dataset in: {default_data_path}")
            train_files = glob.glob(os.path.join(default_data_path, "*.parquet"))

            if not train_files:
                raise FileNotFoundError(f"No .parquet files found in {default_data_path}")

            # 通常加载第一个文件进行检查，或者可以加载所有文件并连接
            # 这里为了简单和快速，我们只加载第一个
            first_file = train_files[0]
            print(f"Loading dataset from: {first_file}")
            dataset = Dataset.from_parquet(first_file)

        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return None


def print_sample(sample_id, dataset):
    """打印单个样本的内容"""
    try:
        if sample_id >= len(dataset):
            print(
                f"Warning: Requested ID {sample_id} is out of range for this dataset portion (size: {len(dataset)}). Skipping.")
            return

        sample = dataset[sample_id]
        # 确保 'translation' 字段存在
        if 'translation' not in sample:
            print(f"Warning: Sample ID {sample_id} does not have a 'translation' field. Content: {sample}")
            return

        translation = sample['translation']
        # 确保 'zh' 和 'en' 字段存在
        zh_text = translation.get('zh', "N/A")
        en_text = translation.get('en', "N/A")

        print(f"--- ID {sample_id} ---")
        print(f"  中文: {zh_text}")
        print(f"  英文: {en_text}")
        print("-" * 20)
    except Exception as e:
        print(f"Error printing sample ID {sample_id}: {e}", file=sys.stderr)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Check the content of specific samples in the WMT19 zh-en dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/check_sensitive_sample.py 12
  python scripts/check_sensitive_sample.py 12 15 20
  python scripts/check_sensitive_sample.py --file data/raw/wmt19_zh_en/train/train-00001-of-00013.parquet 100 200
        """
    )
    parser.add_argument(
        'ids',
        metavar='ID',
        type=int,
        nargs='+',
        help='One or more integer IDs of the samples to check.'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to a specific .parquet file to load instead of the default first file.',
        default=None
    )

    args = parser.parse_args()
    ids_to_check = args.ids
    parquet_file_path = args.file

    dataset = load_dataset(parquet_file_path)
    if dataset is None:
        sys.exit(1)  # 退出码1表示加载失败

    print(f"Dataset loaded successfully. Total samples available in this portion: {len(dataset)}")
    print("=" * 30)

    for sid in ids_to_check:
        print_sample(sid, dataset)

    print("Finished checking samples.")


if __name__ == "__main__":
    main()