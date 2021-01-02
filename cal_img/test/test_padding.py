from utils import cutout
from utils import padding

import cv2

path = "D://data//origin//train_831//B2F//00012.jpg"
img = cutout.cut_out_pic(path)
padding_img = padding.pad_to(img, 160, 360)

cv2.imwrite("D://5-padding_img.jpg", padding_img)
