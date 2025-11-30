# ONNX PaddleOCR

这是一个基于 ONNX 的 PaddleOCR 实现，支持文字检测和识别功能。

## 特性

- 支持多种语言文字识别
- 基于 ONNX，无需安装 PaddlePaddle
- 支持 GPU 加速（需要 onnxruntime-gpu）
- 轻量级部署

## 安装

```bash
pip install onnx-paddleocr
```

## 使用方法

### 命令行使用

```bash
ocr image.jpg
```

### Python API 使用

```python
from onnx_ocr import ONNXPaddleOcr
import cv2

# 初始化OCR
ocr = ONNXPaddleOcr()

# 读取图片
img = cv2.imread('image.jpg')

# 执行OCR
result = ocr.ocr(img)

# 输出结果
for line in result[0]:
    print(f"文本: {line[1][0]}, 置信度: {line[1][1]}")
```

## 模型文件

由于 PyPI 对包大小的限制（100MB），模型文件不会包含在安装包中。首次运行时程序会自动从 GitHub Releases 下载所需模型文件到本地 [models](file:///D:/github/meme2046/onnx-paddleocr/models) 目录。

如果您希望手动下载模型，请访问以下链接：
- [检测模型](https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/det.onnx)
- [识别模型](https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/rec.onnx)
- [分类模型](https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/cls.onnx)
- [字符字典](https://github.com/meme2046/onnx-paddleocr/releases/download/v0.1.0/ppocrv5_dict.txt)

将这些文件放置在如下目录结构中：
```
models/
└── ppocrv5_server/
    ├── det/
    │   └── det.onnx
    ├── rec/
    │   └── rec.onnx
    ├── cls/
    │   └── cls.onnx
    └── ppocrv5_dict.txt
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --use-angle-cls | False | 是否使用角度分类器 |
| --use-gpu | False | 是否使用GPU |
| --det-model | ./models/ppocrv5_server/det/det.onnx | 检测模型路径 |
| --rec-model | ./models/ppocrv5_server/rec/rec.onnx | 识别模型路径 |
| --cls-model | ./models/ppocrv5_server/cls/cls.onnx | 分类模型路径 |
| --dict-path | ./models/ppocrv5_server/ppocrv5_dict.txt | 字符字典路径 |

## 许可证

本项目采用 MIT 许可证。