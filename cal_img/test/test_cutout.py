from utils import cutout

import cv2

path = "D://data//origin//train_831//B2F//00012.jpg"

img = cutout.cut_out_pic(path)

# cv2.imwrite("D://test.jpg", img)
