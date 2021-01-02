from utils.read_img import get_gray_img
import cv2

path = "D://data//origin//train_831//B2F//00012.jpg"

img = get_gray_img(path)
# ret6, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("test", img)
cv2.waitKey(0)
print(type(img))

# cv2.imwrite("D://1.jpg", img)
