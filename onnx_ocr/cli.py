# cli.py
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import typer

from onnx_ocr.onnx_paddleocr import ONNXPaddleOcr
from utils.helper import process_bounding_box

app = typer.Typer(help="onnx paddleocr 命令行工具")


@app.command()
def ocr(
    image_path: str = typer.Argument(..., help="输入图像路径"),
    output_path: Optional[str] = typer.Option(
        "./output", "--output", "-o", help="输出结果文件路径(JSON格式)"
    ),
    use_angle_cls: bool = typer.Option(
        False, "--use-angle-cls", help="是否使用角度分类器"
    ),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="是否使用GPU"),
    det_model_dir: str = typer.Option("", "--det-model", help="检测模型路径"),
    rec_model_dir: str = typer.Option("", "--rec-model", help="识别模型路径"),
    models_dir: str = typer.Option(
        None, "--models-dir", help="模型目录路径，默认为 ~/.onnx/models/"
    ),
    auto_download: bool = typer.Option(
        True, "--auto-download/--no-auto-download", help="是否自动下载模型"
    ),
    cls_model_dir: str = typer.Option(
        "./models/ppocrv5_server/cls/cls.onnx", "--cls-model", help="分类模型路径"
    ),
    rec_char_dict_path: str = typer.Option(
        "./models/ppocrv5_server/ppocrv5_dict.txt", "--dict-path", help="字符字典路径"
    ),
    det_limit_side_len: float = typer.Option(
        960, "--det-limit-len", help="检测图像边长限制"
    ),
    det_db_thresh: float = typer.Option(0.3, "--det-db-thresh", help="DB检测阈值"),
    det_db_box_thresh: float = typer.Option(
        0.6, "--det-db-box-thresh", help="DB检测框阈值"
    ),
    det_db_unclip_ratio: float = typer.Option(
        1.5, "--det-db-unclip-ratio", help="DB检测框放大比例"
    ),
    rec_image_shape: str = typer.Option(
        "3, 48, 320", "--rec-image-shape", help="识别图像形状"
    ),
    drop_score: float = typer.Option(0.5, "--drop-score", help="文本置信度阈值"),
    cpu_threads: int = typer.Option(16, "--cpu-threads", help="CPU推理线程数"),
    cls_thresh: float = typer.Option(0.9, "--cls-thresh", help="分类器阈值"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细信息"),
):
    """
    对指定图像执行OCR识别
    """
    # 获取默认模型目录
    if not models_dir:
        from onnx_ocr.model_loader import get_default_models_dir

        models_dir = get_default_models_dir()

    # 设置默认模型路径
    if not det_model_dir:
        det_model_dir = os.path.join(models_dir, "ppocrv5_server/det/det.onnx")
    if not rec_model_dir:
        rec_model_dir = os.path.join(models_dir, "ppocrv5_server/rec/rec.onnx")

    # 自动下载模型文件
    if auto_download:
        from onnx_ocr.model_loader import download_models_if_needed

        download_models_if_needed(models_dir)

    # 检查输入文件是否存在
    if not Path(image_path).exists():
        typer.echo(f"错误: 图像文件 '{image_path}' 不存在", err=True)
        raise typer.Exit(code=1)

    # 初始化OCR模型
    model_params = {
        "use_angle_cls": use_angle_cls,
        "use_gpu": use_gpu,
        "det_model_dir": det_model_dir,
        "rec_model_dir": rec_model_dir,
        "cls_model_dir": cls_model_dir,
        "rec_char_dict_path": rec_char_dict_path,
        "det_limit_side_len": det_limit_side_len,
        "det_db_thresh": det_db_thresh,
        "det_db_box_thresh": det_db_box_thresh,
        "det_db_unclip_ratio": det_db_unclip_ratio,
        "rec_image_shape": rec_image_shape,
        "drop_score": drop_score,
        "cpu_threads": cpu_threads,
        "cls_thresh": cls_thresh,
    }

    if verbose:
        typer.echo("正在初始化OCR模型...")

    try:
        model = ONNXPaddleOcr(**model_params)
    except Exception as e:
        typer.echo(f"模型初始化失败: {e}", err=True)
        raise typer.Exit(code=1)

    # 读取图像
    if verbose:
        typer.echo(f"正在读取图像: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        typer.echo(f"错误: 无法读取图像文件 '{image_path}'", err=True)
        raise typer.Exit(code=1)

    # 执行OCR
    if verbose:
        typer.echo("正在执行OCR识别...")

    try:
        start_time = time.perf_counter()
        result = model.ocr(img)
        ocr_time = time.perf_counter() - start_time
    except Exception as e:
        typer.echo(f"OCR识别失败: {e}", err=True)
        raise typer.Exit(code=1)

    # 处理结果
    ocr_results = [
        {
            "text": line[1][0],
            "confidence": float(line[1][1]),
            "bounding_box": process_bounding_box(line[0]),
        }
        for line in result[0]
    ]

    # 输出结果
    if output_path:
        # 保存到文件
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{Path(image_path).stem}_rec.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            typer.echo(f"结果已保存到: {output_path}, 识别耗时: {ocr_time:.4f}秒")
        except Exception as e:
            typer.echo(f"保存结果失败: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        # 直接打印到控制台
        for item in ocr_results:
            typer.echo(f"文本: {item['text']}, 置信度: {item['confidence']:.4f}")

        if verbose:
            typer.echo(f"\n总共识别到 {len(ocr_results)} 个文本框")


# @app.command()
# def version():
#     """
#     显示版本信息
#     """
#     project = read_toml("./pyproject.toml")["project"]
#     typer.echo(f"v{project['version']}")
