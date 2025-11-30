import cv2

from onnx_ocr import ONNXPaddleOcr, process_bounding_box

# 初始化OCR
model = ONNXPaddleOcr(
    use_angle_cls=False,
    use_gpu=False,
    cpu_threads=16,
    det_model_dir="",  # 不指定默认:~/.onnx/models/ppocrv5_server/det/det.onnx
    rec_model_dir="",  # 不指定默认:~/.onnx/models/ppocrv5_server/rec/rec.onnx
)

# 读取图片
img = cv2.imread("./images/test.jpg")

# 执行OCR
result = model.ocr(img)

# 输出结果
ocr_results = [
    {
        "text": line[1][0],
        "confidence": float(line[1][1]),
        "bounding_box": process_bounding_box(line[0]),
    }
    for line in result[0]
]
