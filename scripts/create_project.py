import os

# 项目结构定义
STRUCTURE = {
    "config": ["model_config.json", "train_config.yaml"],
    "data": {
        "raw": [],
        "processed": [],
        "teacher_logits": []
    },
    "models": ["__init__.py", "tiny_transformer.py", "tokenizer_wrapper.py"],
    "scripts": ["download_data.py", "generate_logits.py", "quantize_model.py"],
    "src": ["__init__.py", "train_distill.py", "loss.py", "evaluate.py"],
    "tests": ["test_model.py"],
    "outputs": {
        "checkpoints": [],
        "logs": [],
        "reports": []
    }
}

# 根文件
ROOT_FILES = [".gitignore", "requirements.txt", "README.md"]

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)
        if isinstance(content, list):
            for file in content:
                open(os.path.join(path, file), 'a').close()
        elif isinstance(content, dict):
            create_structure(path, content)

# 创建项目
project_name = "rk_llm_distill"
os.makedirs(project_name, exist_ok=True)

# 创建子目录和文件
create_structure(project_name, STRUCTURE)

# 创建根文件
for file in ROOT_FILES:
    open(os.path.join(project_name, file), 'a').close()

print(f"✅ 项目 {project_name} 创建完成！")
print(f"📁 位置: {os.path.abspath(project_name)}")