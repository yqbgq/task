from utils.read_img import get_gray_img
from utils import conv
from utils import crop
import numpy as np
import cv2


def test_edge_cutting(laplace_image: np.ndarray, image: np.ndarray):
    return crop.edge_cutting(laplace_image, image)


def test_crop_center(image: np.ndarray, x=50, y=50):
    return crop.crop_center(image, x, y)


if __name__ == "__main__":
    path = "D://data//origin//train_831//B2F//00012.jpg"
    img = get_gray_img(path)

    # sobel_img = sobel.sobel_cal(img, open_cv=True)
    #
    # Laplace = [
    #     [1, 1, 1],
    #     [1, 8, 1],
    #     [1, 1, 1]
    # ]

    # laplace_model = np.asarray(Laplace)
    #
    # laplace_img = conv.conv2D(img, model=laplace_model, binary=True, open_cv=True)  # type: np.ndarray
    #
    # cropped_img = test_edge_cutting(laplace_img, sobel_img)  # type: np.ndarray
    # cropped_img = cropped_img.astype(np.uint8)
    #
    # # cv2.imshow("test", cropped_img)
    # i = test_crop_center(cropped_img)
    # cv2.imshow("test", i)
    # cv2.waitKey(0)
