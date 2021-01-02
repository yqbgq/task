import cv2
import os

test_img_path = "D://data//processed//test//"
img_path = "D://data//processed//train//"
max_x = 0
max_y = 0

for path in os.listdir(img_path):
    img = cv2.imread(os.path.join(img_path, path), cv2.IMREAD_GRAYSCALE)
    x = img.shape[1]
    y = img.shape[0]
    if x > max_x:
        max_x = x
    if y > max_y:
        max_y = y

for path in os.listdir(test_img_path):
    img = cv2.imread(os.path.join(test_img_path, path), cv2.IMREAD_GRAYSCALE)
    x = img.shape[1]
    y = img.shape[0]
    if x > max_x:
        max_x = x
    if y > max_y:
        max_y = y

print(max_x, max_y)