import os
import urllib.request
from pathlib import Path


def download_models_if_needed(models_dir="./models"):
    """
    如果模型文件不存在,则从预定义的URL下载模型文件
    """
    # 确保模型目录存在
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # 定义模型文件和对应的下载链接
    model_files = {
        "ppocrv5_server/det/det.onnx": "https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/det.onnx",
        "ppocrv5_server/rec/rec.onnx": "https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/rec.onnx",
    }

    # 检查并下载每个模型文件
    for rel_path, url in model_files.items():
        full_path = os.path.join(models_dir, rel_path)
        # 确保目录存在
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(full_path):
            print(f"正在下载 {rel_path}...")
            urllib.request.urlretrieve(url, full_path)
            print(f"{rel_path} 下载完成")
