import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import cv2

"""
第四部分：核心部分，模型的实现以及识别覆盖
"""


# 霍夫线检测(直线)
def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)


def detect_lane_lines(image):
    # 加载预训练的DeepLabV3+模型
    model = deeplabv3_resnet101(pretrained=True)
    model = model.eval()

    # 加载图像并转换为RGB格式
    transform = transforms.Compose([
        transforms.Resize((520, 520)),  # 根据需要调整输入图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化参数
    ])
    image = transform(image).unsqueeze(0)  # 添加批处理维度

    # 运行模型进行车道线检测
    with torch.no_grad():
        output = model(image)["out"][0]
        output = F.softmax(output, dim=0)
        output = torch.argmax(output, dim=0)  # 获取最大概率的类别标签
        output = output.cpu().numpy()  # 转换为numpy数组
        output = (output == 3).astype(np.uint8)  # 将类别标签为3（车道线）的像素设置为1，其他设置为0

    # 将输出转换为二值图像（车道线为白色，背景为黑色）
    output = cv2.resize(output, (image.shape[2], image.shape[1]))  # 调整大小以匹配原始图像大小
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)  # 使用JET颜色映射将输出转换为彩色图像，便于可视化车道线位置

    # 提取轮廓
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return output


# list_of_lines = list(map(hough_lines, roi_images))

# 车道线合理性判断，以斜率把一个图片的线分为两类，一类是左边的线，一类是右边的线。
def average_slope_intercept(lines):
    if lines is None:
        return None
    left_lines = []  # (slope, intercept)   斜率、截距 y = ax + b
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:  # 忽略垂直线
                continue
            slope = (y2 - y1) / (x2 - x1)  # tan@ = y/x = (y2-y1)/(x2-x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)  # 两点之间的距离
            if slope >= 0:
                right_lines.append((slope, intercept))
                right_weights.append(length)
            else:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append(length)

    # 增加线条的权重
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


# 确保直线端点坐标为整型(像素值)
def make_line_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line

    # 确保坐标点为整型值，因为 cv2.line 需要
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return (x1, y1), (x2, y2)


# 车道线检测函数-根据直线的斜率延长车道线的长度
def lane_lines(image, lines):
    if lines is None:
        return []
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]  # bottom of image
    y2 = y1 * 0.6
    # 转换成整型值的坐标
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line  # 延长后直线的新的端点坐标


# 绘制车道线
def draw_lane_lines(image, lines, color=(255, 0, 0), thichness=20):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:  # *line  =>  *((x1, y1), (x2, y2))
            cv2.line(line_image, *line, color, thichness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0)


# lane_images=[]
# for image, lines in zip(test_images, list_of_lines):
#     lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))
#
# show_images(lane_images)

