import os
import json
import time
import logging
from tqdm import tqdm
import dashscope
from datasets import Dataset
import glob
import threading
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 配置 DashScope API Key（请替换为你的实际 Key）
dashscope.api_key = "sk-b0c78a77e5ea489b8c68e0b5049204c6"  # 👈 替换为你的真实 API Key！

# ✅ 全局加载 Qwen-1.5-1.8B（只加载一次，避免重复加载）
QWEN_MODEL = None
QWEN_TOKENIZER = None

# ✅ 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generate_logits.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def load_qwen_model():
    """加载 Qwen-1.5-1.8B 模型（本地）"""
    global QWEN_MODEL, QWEN_TOKENIZER

    if QWEN_MODEL is None or QWEN_TOKENIZER is None:
        # ✅ 获取项目根目录（绝对路径）
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ✅ 构建本地模型路径（确保是目录，不是文件）
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen_tokenizer_offline", "qwen", "Qwen1___5-1___8B")

        # ✅ 检查目录是否存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ 本地模型目录不存在: {MODEL_PATH}")

        # ✅ 检查关键文件是否存在
        required_files = ["config.json", "tokenizer_config.json", "vocab.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(MODEL_PATH, file)):
                raise FileNotFoundError(f"❌ 本地模型缺少关键文件: {file}")

        print(f"📥 正在加载 Qwen-1.5-1.8B 模型（本地）: {MODEL_PATH}")

        try:
            # ✅ 强制离线加载（避免联网）
            QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,  # 👈 强制离线加载
                trust_remote_code=True
            )
            QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,  # 👈 强制离线加载
                trust_remote_code=True
            )
            QWEN_MODEL.eval()
            print("✅ Qwen-1.5-1.8B 模型加载成功！")
        except Exception as e:
            print(f"❌ Qwen-1.5-1.8B 模型加载失败: {str(e)}")
            raise


def call_qwen_translate_local(text, target_lang="en", max_retries=3):
    """调用本地 Qwen-1.5-1.8B 翻译"""
    global QWEN_MODEL, QWEN_TOKENIZER
    load_qwen_model()  # 确保模型已加载

    for attempt in range(max_retries):
        try:
            # 构造 Prompt（与 DashScope API 一致）
            source_lang_full = "中文" if target_lang == "en" else "英文"
            target_lang_full = "英文" if target_lang == "en" else "中文"
            prompt = (
                f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
                f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
                f"只输出翻译结果，不要添加任何解释或额外内容：\n\n"
                f"{text}"
            )

            # 编码输入
            inputs = QWEN_TOKENIZER(prompt, return_tensors="pt").to(QWEN_MODEL.device)

            # 推理（半精度）
            with torch.no_grad():
                outputs = QWEN_MODEL.generate(
                    **inputs,
                    max_new_tokens=512,  # 最大生成长度
                    temperature=0.1,  # 低温度，确保翻译一致性
                    do_sample=False,  # 贪心解码
                    top_k=1,
                    top_p=0.9
                )

            # 解码输出
            generated = outputs[0][inputs.input_ids.shape[1]:]  # 去掉输入部分
            translated_text = QWEN_TOKENIZER.decode(generated, skip_special_tokens=True).strip()

            return translated_text

        except Exception as e:
            logging.warning(f"⚠️  本地翻译失败 (尝试 {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def call_qwen_translate_worker(local_task_queue, local_result_queue):
    """
    多线程翻译工作线程，使用传入的队列。
    通过调用 call_qwen_translate 来实现翻译和回退逻辑。
    """
    while True:
        task = local_task_queue.get()
        if task is None:
            local_task_queue.task_done()
            break

        text, target_lang, idx = task
        try:
            # --- 关键修改：调用你封装好的函数 ---
            translated_text = call_qwen_translate(text=text, target_lang=target_lang)

            if translated_text:
                # 成功，将结果放入结果队列
                local_result_queue.put((idx, translated_text, None))
            else:
                # 函数返回 None，表示翻译失败
                local_result_queue.put((idx, None, "Translation returned None or failed after retries."))

        except Exception as e:
            # 捕获 call_qwen_translate 内部未处理的异常或传递过程中的任何问题
            local_result_queue.put((idx, None, f"Worker caught exception: {e}"))
        finally:
            local_task_queue.task_done()


def call_qwen_translate_api(text, target_lang="en", max_retries=3):
    """
    调用 Qwen-Max 翻译，带重试机制和优化后的逻辑。
    主动检查 TextTranslation 是否存在，以决定调用路径。
    """

    # --- 优化点：主动检查 TextTranslation 是否存在 ---
    # 使用 hasattr 进行检查，避免不必要的 AttributeError 异常
    TEXT_TRANSLATION_AVAILABLE = hasattr(dashscope, 'TextTranslation')
    # --- 优化点结束 ---

    for attempt in range(max_retries):
        logging.info(f"Starting translation attempt {attempt + 1}")

        # --- 根据检查结果，选择调用路径 ---
        if TEXT_TRANSLATION_AVAILABLE:
            # 如果 TextTranslation 存在，则尝试使用它
            logging.debug(f"Attempt {attempt + 1}: TextTranslation is available, using it.")
            try:
                response = dashscope.TextTranslation.call(
                    model='qwen-max',
                    text=text,
                    target_lang=target_lang
                )
                logging.debug("TextTranslation API call made.")

                if response.status_code == 200:
                    logging.debug("TextTranslation API call successful.")
                    return response.output['translated_text']
                else:
                    logging.warning(f"⚠️  TextTranslation API 调用失败 (尝试 {attempt + 1}): {response.message}")

            except Exception as specific_api_error:  # 捕获 TextTranslation API 调用过程中的所有其他异常
                logging.warning(f"⚠️  TextTranslation API 调用异常 (尝试 {attempt + 1}): {specific_api_error}")

        else:
            # 如果 TextTranslation 不存在，或者首选 TextTranslation 失败，则使用 Generation API
            # 注意：这里的逻辑现在变成了“首选 Generation”而不是“回退到 Generation”
            logging.debug(f"Attempt {attempt + 1}: TextTranslation not available or preferred, using Generation API.")

            try:
                # 构造 Prompt，明确指示源语言和目标语言
                source_lang_full = "中文" if target_lang == "en" else "英文"
                target_lang_full = "英文" if target_lang == "en" else "中文"
                prompt = (
                    f"你是一位精通{source_lang_full}和{target_lang_full}的专业翻译人员。\n"
                    f"请将以下{source_lang_full}文本准确、自然地翻译成{target_lang_full}，"
                    f"只输出翻译结果，不要添加任何解释或额外内容：\n\n"
                    f"{text}"
                )

                response = dashscope.Generation.call(
                    model='qwen-max',
                    prompt=prompt
                    # 可根据需要添加其他参数，例如 max_tokens, temperature 等
                )
                logging.debug("Generation API call made.")

                if response.status_code == 200:
                    # Generation API 返回的文本通常在 'text' 或 'output' 字段
                    # 具体取决于响应格式，这里假设是 'text'
                    translated_text = response.output.get('text', '').strip()
                    if translated_text:
                        logging.debug("Generation API call successful.")
                        return translated_text
                    else:
                        logging.warning(f"⚠️  Generation API returned empty text (尝试 {attempt + 1})")
                else:
                    logging.warning(f"⚠️  Generation API 调用失败 (尝试 {attempt + 1}): {response.message}")

            except Exception as general_api_error:  # 捕获 Generation API 调用过程中的所有异常
                logging.warning(f"⚠️  Generation API 调用异常 (尝试 {attempt + 1}): {general_api_error}")

        # --- 重试逻辑 ---
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 指数退避
            logging.info(f"⏳ 等待 {wait_time} 秒后进行下一次尝试...")
            time.sleep(wait_time)

    # 所有尝试都失败后，返回 None
    logging.error(f"❌ 经过 {max_retries} 次尝试后，所有翻译方法均失败。")
    return None

def call_qwen_translate(text, target_lang="en", max_retries=3):
    """调用 Qwen 翻译（优先本地，失败回退 API）"""
    # ✅ 优先调用本地模型
    translated = call_qwen_translate_local(text, target_lang, max_retries)
    if translated:
        return translated

    # ❌ 回退到 DashScope API（如果本地失败）
    logging.warning("⚠️  本地翻译失败，回退到 DashScope API")
    return call_qwen_translate_api(text, target_lang, max_retries)  # 你原来的 API 调用函数

# 相应地，修改 generate_teacher_logits 函数，移除其中的线程启动和停止逻辑
def generate_teacher_logits(
        local_dataset_path="data/raw/wmt19_zh_en",
        output_dir="data/teacher_logits",
        max_samples=None,  # None = 全量生成
        start_from=0,
        # num_threads=4  # 不再需要作为参数传递给此函数，可以在 generate_direction_multithread 内部定义或传递
):
    """生成教师模型软标签（中英互译）"""
    try:
        end_sample_info = "全部" if max_samples is None else str(start_from + max_samples)
        logging.info(f"🧠 开始生成教师翻译 (样本 {start_from} 到 {end_sample_info})...")

        # ✅ 获取项目根目录（绝对路径）
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ABS_LOCAL_DATASET_PATH = os.path.join(PROJECT_ROOT, local_dataset_path)

        # ✅ 检查目录是否存在
        if not os.path.exists(ABS_LOCAL_DATASET_PATH):
            raise FileNotFoundError(f"❌ 数据集目录不存在: {ABS_LOCAL_DATASET_PATH}")

        os.makedirs(output_dir, exist_ok=True)

        # ✅ 从本地 .parquet 文件加载数据（直接加载第一个文件）
        train_files = glob.glob(os.path.join(ABS_LOCAL_DATASET_PATH, "train", "*.parquet"))
        if not train_files:
            raise FileNotFoundError(f"❌ 未找到训练集文件: {os.path.join(ABS_LOCAL_DATASET_PATH, 'train', '*.parquet')}")

        dataset = Dataset.from_parquet(train_files[0])  # <--- 确保这行存在

        # 设置样本数量
        total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        logging.info(f"📊 总样本数: {total_samples}")

        # 生成中→英数据 (会内部启动和停止线程)
        zh_to_en_file = os.path.join(output_dir, "zh_to_en.jsonl")
        zh_success, zh_fail = generate_direction_multithread(
            dataset=dataset,
            src_lang="zh",
            tgt_lang="en",
            output_file=zh_to_en_file,
            max_samples=total_samples,
            start_from=start_from
        )

        # 生成英→中数据 (会内部启动和停止线程)
        en_to_zh_file = os.path.join(output_dir, "en_to_zh.jsonl")
        en_success, en_fail = generate_direction_multithread(
            dataset=dataset,
            src_lang="en",
            tgt_lang="zh",
            output_file=en_to_zh_file,
            max_samples=total_samples,
            start_from=start_from
        )

        logging.info("✅ 教师软标签生成完成！")
        return zh_success + en_success, zh_fail + en_fail

    except Exception as e:
        logging.error(f"❌ 生成失败: {str(e)}")
        raise  # 重新抛出异常


def generate_direction_multithread(dataset, src_lang, tgt_lang, output_file, max_samples, start_from):
    """
    多线程生成单向翻译数据（中→英 或 英→中）
    """
    # 创建局部队列
    local_task_queue = queue.Queue()
    local_result_queue = queue.Queue()
    num_threads = 4  # 可以作为参数传入，或在此处定义

    # 启动多线程
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=call_qwen_translate_worker, args=(local_task_queue, local_result_queue))
        t.start()
        threads.append(t)

    # 初始化返回值
    success_count = 0
    fail_count = 0

    try:  # 使用 try...finally 确保线程能被正确停止
        # 如果文件已存在，读取已处理样本数
        processed_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for _ in f)
            logging.info(f"📌 已存在 {processed_count} 条记录，从第 {processed_count} 条开始")
            start_from = processed_count

        # 获取数据切片
        end_idx = min(start_from + max_samples, len(dataset))

        # 生成翻译（多线程）
        with open(output_file, 'a', encoding='utf-8') as f:
            # 提交任务到队列
            for i in range(start_from, end_idx):
                translation = dataset[i]['translation']
                if src_lang == "zh":
                    src_text = translation['zh']
                else:
                    src_text = translation['en']
                local_task_queue.put((src_text, tgt_lang, i))

            # 处理结果
            pbar = tqdm(total=end_idx - start_from, desc=f"生成 {src_lang}→{tgt_lang} 翻译", initial=0)
            for _ in range(end_idx - start_from):
                idx, trans_text, error = local_result_queue.get()  # 使用局部队列
                if trans_text:
                    translation = dataset[idx]['translation']
                    if src_lang == "zh":
                        ref_text = translation['en']
                    else:
                        ref_text = translation['zh']

                    result = {
                        "id": idx,
                        "src": dataset[idx]['translation'][src_lang],
                        "ref": ref_text,
                        "hyp": trans_text,
                        "task_id": 0 if src_lang == "zh" else 1,
                        "timestamp": time.time()
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    success_count += 1
                else:
                    logging.warning(f"❌ 翻译失败 (ID {idx}): {error}")
                    fail_count += 1

                pbar.update(1)

            pbar.close()

        logging.info(f"✅ {src_lang}→{tgt_lang} 生成完成！成功: {success_count}, 失败: {fail_count}")

    finally:  # 无论是否发生异常，都确保线程停止
        # 停止工作线程
        for i in range(num_threads):
            local_task_queue.put(None)  # 向局部队列发送停止信号
        for t in threads:
            t.join()  # 等待线程结束

    # 在 finally 块之后返回结果
    return success_count, fail_count


def validate_logits_file(file_path):
    """验证生成的 logits 文件"""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    count += 1
        logging.info(f"✅ 验证通过！共 {count} 条有效记录")
        return count
    except Exception as e:
        logging.error(f"❌ 验证失败: {str(e)}")
        return 0


if __name__ == "__main__":
    # 配置日志以便查看调试信息 (如果需要更详细的信息，可以设置为 logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    os.makedirs("logs", exist_ok=True)

    # 🚀 生成指定数量的样本
    success, fail = generate_teacher_logits(
        max_samples=150,  # None = 全量（约 400 万条）
        # num_threads=8  # 如果需要调整，可以在 generate_direction_multithread 内调整
    )

    # 验证生成文件
    validate_logits_file("data/teacher_logits/zh_to_en.jsonl")
    validate_logits_file("data/teacher_logits/en_to_zh.jsonl")

    # # 配置日志以便查看调试信息 (如果需要更详细的信息，可以设置为 logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)
    #
    # # 测试 1: 中文 -> 英文
    # zh_text = "今天天气真好，我们去公园散步吧。"
    # print("-" * 20)
    # print(f"原文 (中文): {zh_text}")
    # trans_en = call_qwen_translate(zh_text, target_lang="en")
    # print(f"译文 (英文): {trans_en}")
    # print("-" * 20)
    #
    # # 测试 2: 英文 -> 中文
    # en_text = "The weather is nice today, let's go for a walk in the park."
    # print(f"原文 (英文): {en_text}")
    # trans_zh = call_qwen_translate(en_text, target_lang="zh")
    # print(f"译文 (中文): {trans_zh}")
    # print("-" * 20)
