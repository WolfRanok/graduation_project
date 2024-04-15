import tensorflow as tf
import tensorflow
import cv2
import numpy as np
from lib.processing_images import *

# 预训练的Deeplabv3+模型权重
model_path = "deeplabv3_plus_model.h5"
model = tensorflow.keras.load_model(model_path)  # 您可能需要提供custom_objects来处理自定义层

# 如果您使用自定义实现或其他格式，您需要使用适当的方法来加载模型。
DESIRED_WIDTH = 512
DESIRED_HEIGHT = 512


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Deeplab在灰度图上训练
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deeplab在灰度图上训练
    image = convert_gray_scale(image)   # 灰度化
    image = apply_smooothing(image)     # 滤波去噪
    image = image_threshold(image)      # 二值化

    image = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT))
    image = image / 255.0  # 归一化
    return image


image = preprocess_image("data/image.jpg")
image = np.expand_dims(image, axis=0)  # 为模型输入增加批处理维度

predictions = model.predict(image)