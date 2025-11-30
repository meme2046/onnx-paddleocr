# ONNX PaddleOCR

这是一个基于 ONNX 的 PaddleOCR 实现，支持文字检测和识别功能。

## 特性

-   支持多种语言文字识别
-   基于 ONNX，无需安装 PaddlePaddle 和 paddleocr
-   支持 GPU 加速(需要 onnxruntime-gpu)
-   轻量级部署

## 安装

```bash
pip install onnx-paddleocr # pip安装
uv add onnx-paddleocr # uv安装
```

## 使用方法

### 命令行使用

```bash
pipx install onnx-paddleocr # pipx安装
uv tool install onnx-paddleocr # uv安装
ocr --help # 帮助信息
ocr ./images/test.jpg # 示例:识别图片
```

### Python API 使用

```python
from onnx_ocr import ONNXPaddleOcr
import cv2

# 初始化OCR
model = ONNXPaddleOcr(
        use_angle_cls=False,
        use_gpu=False,
        cpu_threads=16,
        det_model_dir="", # 不指定默认:~/.onnx/models/ppocrv5_server/det/det.onnx
        rec_model_dir="", # 不指定默认:~/.onnx/models/ppocrv5_server/rec/rec.onnx
    )

# 读取图片
img = cv2.imread('./images/test.jpg')

# 执行OCR
result = ocr.ocr(img)

# 输出结果
for line in result[0]:
    print(f"文本: {line[1][0]}, 置信度: {line[1][1]}")
```

## 模型文件

模型采用`ppocrv5_server`文件不会包含在安装包中，首次运行时程序会自动从 GitHub Releases 下载所需模型文件到本地用户目录下的：`~/.onnx/models/ppocrv5_server`

如果您希望手动下载模型，请访问以下链接：

-   [检测模型](https://github.com/meme2046/onnx-paddleocr/releases/download/ppocrv5_server/det.onnx)
-   [识别模型](https://github.com/meme2046/onnx-paddleocr/releases/download/ppocrv5_server/rec.onnx)

将这些文件放置在如下目录结构中：

```
~/.onnx/models/
└── ppocrv5_server/
    ├── det/
    │   └── det.onnx
    ├── rec/
    │   └── rec.onnx
    ├── cls/
    │   └── cls.onnx
    └── ppocrv5_dict.txt
```

## 部分参数说明

| 参数            | 默认值                                     | 说明               |
| --------------- | ------------------------------------------ | ------------------ |
| --use-angle-cls | False                                      | 是否使用角度分类器 |
| --use-gpu       | False                                      | 是否使用 GPU       |
| --det-model     | ~/.onnx/models/ppocrv5_server/det/det.onnx | 检测模型路径       |
| --rec-model     | ~/.onnx/models/ppocrv5_server/rec/rec.onnx | 识别模型路径       |

## 许可证

本项目采用 MIT 许可证。
