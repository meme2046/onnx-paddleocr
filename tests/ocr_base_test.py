import logging
import time

import cv2

from onnx_ocr import ONNXPaddleOcr, result_to_json_data
from utils.elapsed import timeit
from utils.logger import get_logger

logger = get_logger(__name__, logging.DEBUG)


@timeit
def test_rec(fp: str = "./images/shijuan.jpg"):
    model = ONNXPaddleOcr(
        use_angle_cls=False,
        use_gpu=False,
        cpu_threads=16,  # CPU推理线程数
        det_model_dir="",
        rec_model_dir="",
    )
    img = cv2.imread(fp)
    s = time.time()
    result = model.ocr(img)
    e = time.time()
    logger.debug("total time: {:.3f}".format(e - s))
    json_data = result_to_json_data(result)

    texts = [item["text"] for item in json_data]
    logger.debug(texts)
    logger.debug(json_data)

    # path_obj = Path(fp)
    # out_fp = str(path_obj.with_name(path_obj.stem + "_rec" + path_obj.suffix))
    # sav2Img(img, result, out_fp)

    # save_to_img(img, result, "output")
    # save_to_json(json_data, "output")


if __name__ == "__main__":
    test_rec("./images/27.jpg")
    # test_rec("./images/shijuan.jpg")
