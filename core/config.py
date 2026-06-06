# core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 明确指定 .env 文件路径（项目根目录）
env_path = Path(__file__).parent.parent/ "config.env"
load_dotenv(dotenv_path=env_path)

# 现在从环境变量读取
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
}

MEM0_CONFIG = {
    "api_key": os.getenv("MEM0_API_KEY"),
}
print("Loading .env from:", env_path)
print("File exists:", env_path.exists())
# 可选：检查必要的配置是否存在
# if not LLM_CONFIG["api_key"]:
#     raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
