from design import *
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QSizePolicy
from line_detection import Ui_MainWindow
from PyQt5.QtGui import QMouseEvent, QIcon, QPixmap
from PyQt5.QtCore import QPoint, Qt
from threading import Thread
import traceback
import re
import sys
from PIL import Image
from ultralytics import YOLO
from lib import video
import cv2


class solve_image:
    def __init__(self):
        self.number = None
        self.model = YOLO('yolov8n.pt')  # 初始化训练好的yolov8模型

    def analysis(self, image_path):
        """
        用于分析图片
        :param image_path:图片路径
        :return:
        """
        self.number = 1
        self.image = cv2.imread(image_path)
        self.gray_image = convert_gray_scale(self.image)  # 灰度图像
        self.blured_image = apply_smooothing(self.gray_image)  # 滤波降噪
        self.binary_image = image_threshold(self.blured_image)  # 二值化
        self.edge_image = detect_edges(self.binary_image)  # 边缘检测
        self.answer = LaneLine_process(self.image)  # 车道线检测结果

        # 文件保存
        self.save_image(self.gray_image)
        self.save_image(self.blured_image)
        self.save_image(self.binary_image)
        self.save_image(self.edge_image)
        self.save_image(self.answer)

        # 这里单独处理障碍物检测的部分
        # 推理

        self.yolo_image(fr'image_analysis\{self.number - 1}.jpg', fr'image_analysis\{self.number}.jpg')  # 分析障碍物

        self.number += 1

    def yolo_image(self, read_url, save_url=None):
        """
        使用yolo模型处理图像信息
        :param read_url: 输入图像地址
        :param save_url: 保存图像地址（默认与原图像地址一致）
        :return: None
        """
        if save_url is None:
            save_url = read_url

        results = self.model(read_url)

        # 打标签并保存
        num = 0  # 记录障碍物数目
        annotated_frame = None
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    num += 1

            annotated_frame = r.plot()

        if annotated_frame is None:
            print('没有检测到结果')
            return

        # 添加标识
        warning = ''
        if num == 0:
            return
        elif num == 1:
            warning = 'Warning: There is 1 obstacle ahead, please avoid it!'
        else:
            warning = f'Warning: There are {num} obstacle ahead, please avoid it!'
        image = annotated_frame.copy()
        cv2.putText(image, warning, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.imwrite(save_url, image)

    def save_image(self, image):
        """
        用于文件的保存
        :return:
        """
        name = rf'image_analysis\{self.number}.jpg'
        self.number += 1

        if image is None:
            return

        cv2.imwrite(name, image)


class MyWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__init_signals()  # 初始化信号
        self.__init_others()  # 初始化其他部分
        self.__init_image()  # 初始化图片处理部分

    def __init_image(self):
        """
        初始化图片处理部分
        :return:
        """
        self.solve_image = solve_image()  # 初始化图片处理器

    def __init_signals(self):
        """
        初始化信号
        :return:
        """
        self.pushButton_8.clicked.connect(self.close_window)  # 连接按钮点击事件与窗口关闭槽函数
        self.pushButton_10.clicked.connect(self.minimizeWindow)  # 最小化窗口
        self.pushButton_9.clicked.connect(self.toggleWindowSize)  # 切换窗口大小

        self.pushButton.clicked.connect(self.openFileDialog)  # “打开文件”按钮
        self.pushButton_2.clicked.connect(self.image_save)  # “保存文件”按钮
        self.pushButton_3.clicked.connect(lambda: self.show_image_in_label('image_analysis/1.jpg'))  # “灰度化”按钮
        self.pushButton_4.clicked.connect(lambda: self.show_image_in_label('image_analysis/2.jpg'))  # “模糊滤波”按钮
        self.pushButton_5.clicked.connect(lambda: self.show_image_in_label('image_analysis/3.jpg'))  # “二值化”按钮
        self.pushButton_6.clicked.connect(lambda: self.show_image_in_label('image_analysis/4.jpg'))  # “边缘检测”按钮
        self.pushButton_7.clicked.connect(lambda: self.show_image_in_label('image_analysis/5.jpg'))  # “车道线检测”按钮
        self.pushButton_11.clicked.connect(lambda: self.show_image_in_label('image_analysis/6.jpg'))  # “障碍检测”按钮
        self.pushButton_12.clicked.connect(lambda: self.handle_video())  # “视频检测”按钮

    def handle_video(self, image_url=video_analysis_url, save_video_url=video_save, fps=24):
        """
        专门用于处理视频的函数
        :param image_url: 中间文件路径
        :param save_video_url: 视频保存路径
        :param fps: 视频的帧率（每秒几帧）
        :return: None
        """

        self.debug(f'请选择需要处理的视频')
        fname = QFileDialog.getOpenFileName(self, 'Open File', r'test_videos')
        self.video_path = fname[0]

        if self.video_path == '':
            self.debug('已取消选择')
            return

        self.debug(f'正在分解视频中:{self.video_path} ')
        video.analysis(read_url=self.video_path)
        self.debug('分解已完成')

        # 这里开始逐帧分析视频
        self.debug('正在分析视频中')

        for root, dirs, files in os.walk('video_analysis'):  # 获取所有的图片
            for file in files:  # 批量获取原分解图片
                image_name = os.path.join(root, file)
                image = cv2.imread(image_name)
                image_line = LaneLine_process(image)  # 车道线检测后的图片
                cv2.imwrite(image_name, image_line)  # 覆盖原图片

                # 再处理障碍物
                self.solve_image.yolo_image(image_name)

        self.debug('视频分析已完成')

        self.debug('正在合成视频')
        video.compound(image_url, save_video_url, fps)
        self.debug(f'视频已合成！{save_video_url}')

        os.startfile(save_video_url + '\\' + 'test.mp4', 'open')  # 打开视频

    def __init_others(self):
        """
        初始化其他部分
        :return:
        """
        self.children = []  # 线程池
        self.setWindowFlag(Qt.FramelessWindowHint)  # 设置无边框

        self.window_size_flag = True  # 监控窗口大小状态
        self.pushButton_9.setIcon(QIcon('ui_images\max.png'))  # 设置最大化图标

    def image_save(self):
        # 用于保存当前屏幕中的图片

        self.debug(f'请选择文件保存位置')
        directory = QFileDialog.getExistingDirectory(self, 'Open File', r'new_images')  # 文件夹选择器
        self.debug(directory)
        if directory == '':
            self.debug('已取消保存')
            return
        image = plt.imread(self.image_path)  # 读取图片
        image_path = fr'{directory}/{re.search(r".*/(.*)", self.image_path)[1]}'  # 计算文件url
        cv2.imwrite(image_path, image)  # 图片保存

        self.debug(f'文件已保存:{image_path}')

    def close_window(self):
        # 处理关闭窗口事件
        self.close()  # 关闭窗口

    def minimizeWindow(self):
        # 最小化窗口
        self.showMinimized()

    def debug(self, text):
        # 用于添加日志信息
        self.textBrowser.append(text)

    def openFileDialog(self):
        # 打开文件对话框
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open File', r'test_images')
            self.image_path = fname[0]
            if self.image_path == '':
                self.debug('已取消选择，请按照需要重新选择')
                return

            self.debug(f'开始解析:{self.image_path} ')
        except:
            self.debug('请先打开文件！')
            return

        try:
            self.children.append(Thread(target=self.solve_image.analysis, args=(self.image_path,)))
            self.children[-1].start()  # 执行图片分析线程
            self.debug(f'解析完成！')

        except Exception as e:
            self.debug(traceback.format_exc())

        self.show_image_in_label(self.image_path)

    def show_image_in_label(self, image_path):
        # 该函数实现图片展示
        image = cv2.imread(image_path)
        h,w = image.shape[:2]

        pixmap = QPixmap(image_path)

        # size = pixmap.size()  # 获得图像大小
        # 以下为动态计算控件的大小

        self.setGeometry(self.pos().x(),self.pos().y(),w, h + 30 + 205)
        self.centralwidget.setFixedSize(w, h + 30 + 205)
        self.frame.setFixedSize(w,30)
        self.frame_2.setFixedSize(w,h)
        self.frame_3.setFixedSize(w,205)
        self.label.resize(w,h)  # 将标签大小调整为图像大小
        self.repaint()


        self.image_path = image_path  # 记录当前屏幕中的图片路径

        self.label.setPixmap(pixmap)



    def toggleWindowSize(self):
        # 由于切换窗口大小
        if self.window_size_flag:  # 窗口处于最大化的状态
            self.showMaximized()  # 最大化
            self.pushButton_9.setText("")  # 不设置文字
            self.pushButton_9.setIcon(QIcon('ui_images\min.png'))  # 显示图标

        else:
            self.showNormal()  # 恢复正常
            self.pushButton_9.setIcon(QIcon('ui_images\max.png'))  # 显示图标

        self.window_size_flag ^= 1  # 切换状态

    # 以下三个函数实现无边框设置
    def mouseMoveEvent(self, event_: QMouseEvent):
        if self.__is_tracking:
            self.__end_pos = event_.pos() - self.__start_pos
            self.move(self.pos() + self.__end_pos)

    def mousePressEvent(self, event_: QMouseEvent):
        if event_.button() == Qt.LeftButton:
            self.__is_tracking = True
            self.__start_pos = QPoint(event_.x(), event_.y())

    def mouseReleaseEvent(self, event_: QMouseEvent):
        if event_.button() == Qt.LeftButton:
            self.__is_tracking = False
            self.__start_pos = None
            self.__end_pos = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())
    # image = cv2.imread('image_analysis/6.jpg')
    # show_image(image)