import cv2
import numpy as np

"""
第三部分：这里用于实现图片处理的工作，实现将图片的灰度化、滤波去噪、二值化、边缘检测
"""


# 灰度化
def convert_gray_scale(image):
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# gray_images = list(map(convert_gray_scale, white_yello_images))
# show_images(gray_images)
# 滤波去噪
def apply_smooothing(image, ksize=15):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


# blured_images = list(map(apply_smooothing, gray_images))
# show_images(blured_images)


# 边缘检测
def detect_edges(image, low_thresh=50, high_thresh=150):
    return cv2.Canny(image, low_thresh, high_thresh)


# edge_images = list(map(detect_edges, blured_images))
# show_images(edge_images)

# 多边形填充+按位与过滤
def filter_region(image, vertices):
    mask = np.zeros_like(image)  # 全0图像，大小与image相同
    if len(mask.shape) == 2:  # 灰度图，只有宽高，通道数默认为1
        cv2.fillPoly(mask, vertices, 255)
    else:  # 多通道图像
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(image, mask)


# 感兴趣区域  Region of Interest   => ROI
def select_region(image):
    rows, cols = image.shape[0:2]  # 得到宽高
    bottom_left = [cols * 0.1, rows * 0.95]  # 左上、左下，右上、右下
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # 定义一个numpy数组，表示选中的区域
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # 保留选中区域像素值，非感兴趣区域变成0
    return filter_region(image, vertices)


def draw_lines(image, lines, color=(255, 0, 0), thickness=2, cp=True):
    if cp:
        image = np.copy(image)  # 在副本中画图，而不会修改原图
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


# for i in range(len(test_images)):
#     draw_lines(test_images[i], list_of_lines[i])

# line_images = []  # 原图RGB    线段坐标
# for image, lines in zip(test_images, list_of_lines):
#     line_images.append(draw_lines(image, lines))
#
# show_images(line_images)
# roi_images = list(map(select_region, edge_images))
# show_images(roi_images)
if __name__ == '__main__':
    pass
