from ultralytics import YOLO
from PIL import Image


# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# 在图像上运行测试
results = model('human_images/1.jpg')  # 返回 Results 对象

# 处理结果
for result in results:
    im_array = result.plot()  # 绘制包含预测结果的BGR numpy数组
    im = Image.fromarray(im_array[, ::-1])  # RGB PIL图像
    im.show()  # 显示图像