from utils.read_img import get_gray_img
from utils import conv
import numpy as np
import cv2

path = "D://data//origin//train_831//B2F//00012.jpg"
img = get_gray_img(path)

Laplace = [
    [1, 1, 1],
    [1, 8, 1],
    [1, 1, 1]
]

laplace_model = np.asarray(Laplace)

laplace_img = conv.conv2D(img, model=laplace_model, binary=True, open_cv=True)  # type: np.ndarray
laplace_img = laplace_img.astype(np.uint8)

cv2.imshow("test", laplace_img)
cv2.waitKey(0)
