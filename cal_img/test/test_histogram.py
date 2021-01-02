from utils.read_img import get_gray_img
import matplotlib.pyplot as plt
from utils import conv
from utils.histogram import histogram_gray_cal
import time

if __name__ == "__main__":
    path = "D://data//processed//train_831//B2F//00012.jpg"
    img = get_gray_img(path)

    start = time.time()
    result = histogram_gray_cal(img, np_opt=False)
    print(time.time() - start)

    start = time.time()
    result2 = histogram_gray_cal(img, np_opt=True)
    print(time.time() - start)

    print("OK")
