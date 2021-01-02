import time


def print_info(info):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(now, info)
