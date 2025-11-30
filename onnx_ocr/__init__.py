from onnx_ocr.onnx_paddleocr import ONNXPaddleOcr
from utils.toml import read_toml

project = read_toml("./pyproject.toml")["project"]
__version__ = project["version"]

__all__ = [
    ONNXPaddleOcr,
]
