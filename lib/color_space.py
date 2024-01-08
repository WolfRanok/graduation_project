import numpy as np
import cv2


"""
第二部分：色彩空间的选择用于比较色彩空间
"""


# RGB颜色空间
def select_rgb_white_yellow(image):
    lower = np.uint8([200, 200, 200])  # 定义白线的上下界
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    lower = np.uint8([190, 190, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # 合并
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    # 提取白色与黄色
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


# show_images(list(map(select_rgb_white_yellow, test_images)))

# HSV颜色空间
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


# show_images(list(map(convert_hsv, test_images)))

# HSL颜色空间    --黄色和白线都可显示
def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

# show_images(list(map(convert_hsl, test_images)))



# 使用HSL颜色空间对图片进行特定颜色过滤
def select_hsl_white_yellow(image):
    hsl = convert_hsl(image)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(hsl, lower, upper)
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40,255,255])
    yellow_mask = cv2.inRange(hsl, lower, upper)
    # 合并二值图
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    # 二值图与原图进行按位与，得到黄线与白线
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


if __name__ == '__main__':
    pass
    # test_images = [plt.imread(path) for path in glob.glob("../test_images/*.jpg")]
    # # 保存HSL检测结果                                       原始图片
    # white_yello_images = list(map(select_hsl_white_yellow, test_images))
    # show_images(white_yello_images)