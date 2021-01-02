import numpy as np


def pad_to(img: np.ndarray, target_width, target_height):
    img_width = img.shape[1]
    img_height = img.shape[0]

    needed_width = target_width - img_width
    needed_height = target_height - img_height

    pad_row_img = np.row_stack([np.zeros([needed_height // 2, img_width]), img])

    if needed_height % 2 == 0:
        pad_row_img = np.row_stack([pad_row_img, np.zeros([needed_height // 2, img_width])])
    else:
        pad_row_img = np.row_stack([pad_row_img, np.zeros([needed_height // 2 + 1, img_width])])

    pad_col_img = np.column_stack([pad_row_img, np.zeros([target_height, needed_width // 2])])

    if needed_width % 2 == 0:
        pad_col_img = np.column_stack([np.zeros([target_height, needed_width // 2]), pad_col_img])
    else:
        pad_col_img = np.column_stack([np.zeros([target_height, needed_width // 2 + 1]), pad_col_img])

    return pad_col_img
