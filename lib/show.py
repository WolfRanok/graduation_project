import glob
import matplotlib.pyplot as plt
import cv2
"""
第一部分：图片的展示，不包括图片
"""


# 展示单一图片
def show_image(image,name='img'):
    cv2.imshow('img', image)
    cv2.waitKey(0)


# 辅助显示若干图片
def show_images(images, cmap=None):
    if len(images) > 10:  # 只展示前10张图片
        images = images[:10]
    cols = 2
    rows = len(images) // cols  # 乘除  N行2列
    plt.figure(figsize=(10, 11))  # 定义显示图
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)  # 子图像 （行。列。下标）
        cmap = 'gray' if len(image.shape) == 2 else cmap
        #         print(cmap)
        plt.imshow(image, cmap=cmap)  # 灰度或彩色显示
        plt.xticks()  # 不显示刻度
        plt.yticks()
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)  # 紧凑显示
    plt.show()


if __name__ == '__main__':
    test_images = [plt.imread(path) for path in glob.glob("../test_images/*.jpg")]  # rgb的图
    # show_images(test_images)
    show_image(test_images[0])
