import logging
import os
from pathlib import Path

import onnxruntime as ort

from utils.logger import get_logger

logger = get_logger(
    __name__,
    level=logging.DEBUG,
)

if __name__ == "__main__":
    logger.debug(f"ONNX Runtime版本:{ort.__version__}")
    available_providers = ort.get_available_providers()
    logger.debug("可用的执行提供者:")
    for provider in available_providers:
        logger.debug(f"  - {provider}")

    if "CUDAExecutionProvider" in available_providers:
        logger.debug("\n✓ onnxruntime-gpu可用")

        cuda_version = ort.get_device()
        logger.debug(f"ONNX Runtime设备类型: {cuda_version}")
    else:
        print("\n✗ onnxruntime-gpu不可用")

    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_path:
        logger.debug(f"\n✓ 检测到CUDA_PATH环境变量: {cuda_path}")
    else:
        print("\n? 未找到CUDA_PATH环境变量")
        print("  如果您已安装CUDA,可能需要将其添加到系统PATH中")

    sess_options = ort.SessionOptions()
    onnx_path = str(Path.home() / ".onnx/models/ppocrv5_server/rec/rec.onnx")
    try:
        session = (
            ort.InferenceSession(
                onnx_path,
                sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" in available_providers
            else None
        )

        if session:
            actual_providers = session.get_providers()
            if "CUDAExecutionProvider" in actual_providers:
                print("✓ CUDA提供者初始化成功")
        else:
            print("? CUDA提供者不可用")
    except Exception as e:
        print(f"✗ CUDA提供者初始化失败: {e}")
