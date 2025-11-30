import os
import sys
import urllib.request
from pathlib import Path


def get_default_models_dir():
    """获取默认模型目录路径"""
    if sys.platform.startswith("win"):
        # Windows: C:\Users\用户名\.onnx\models\
        home_dir = os.environ.get("USERPROFILE", "")
        if not home_dir:
            home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".onnx", "models")
    else:
        # Unix-like systems: ~/.onnx/models/
        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".onnx", "models")

    return models_dir


def get_default_model_paths(models_dir=None):
    """获取默认模型路径"""
    if models_dir is None:
        models_dir = get_default_models_dir()

    det_model_dir = os.path.join(models_dir, "ppocrv5_server/det/det.onnx")
    rec_model_dir = os.path.join(models_dir, "ppocrv5_server/rec/rec.onnx")

    return {"det_model_dir": det_model_dir, "rec_model_dir": rec_model_dir}


def download_models(det_model_dir, rec_model_dir):
    model_files = {
        det_model_dir: "https://github.com/meme2046/onnx-paddleocr/releases/download/ppocrv5_server/det.onnx",
        rec_model_dir: "https://github.com/meme2046/onnx-paddleocr/releases/download/ppocrv5_server/rec.onnx",
    }

    for rel_path, url in model_files.items():
        Path(rel_path).parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(rel_path):
            print(f"正在从github下载onnx模型: {rel_path}...")
            urllib.request.urlretrieve(url, rel_path)
            print(f"{rel_path} 下载完成")
