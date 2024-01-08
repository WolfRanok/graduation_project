from lib.algorithm import *
from lib.processing_images import *
from lib.show import *
from lib.video import *
from lib.color_space import *


# 车道线检测流程总函数
def LaneLine_process(image):
    test_image = image  # 取一张图片
    white_yello_images = select_hsl_white_yellow(test_image)  # 取黄白部分
    gray_images = convert_gray_scale(white_yello_images)  # 灰度化
    blured_images = apply_smooothing(gray_images)  # 滤波去噪
    edge_images = detect_edges(blured_images)  # 边缘检测
    roi_images = select_region(edge_images)  # 获取感兴趣区域
    list_of_lines = hough_lines(roi_images)  # 获得车道线坐标
    # 绘制车道线
    lane_images = draw_lane_lines(test_image, lane_lines(test_image, list_of_lines),color=(0, 0, 255))
    return lane_images


def operate_image():
    """
    处理test_images下的图片，并作出识别，最终给出结果列表
    :return:list(numpy)
    """
    images_path = [path for path in glob.glob("test_images/*.jpg")]  # 获取所有的jpg图片路径
    original_images = [plt.imread(path) for path in images_path]
    images = [LaneLine_process(image) for image in original_images]
    # show_images(images)
    return images


def operate_video(path=None):
    """
    读取所给路径下的视频或者读取text_videos下的视频，并作出识别，最终结果视频将存放于new_videos文件夹下
    :return: NULL
    """
    if path is None:
        path = get_mp4_files()[0]  # 一次处理一个视频
    image_list = split_video_into_images(path)  # 将视频处理成图片组
    images = [LaneLine_process(image) for image in image_list]  # 逐一处理图片
    ndarray_list_to_video(images)   # 将处理好的图片组装成视频



if __name__ == '__main__':
    operate_video()
