import os
import cv2
from moviepy.editor import ImageSequenceClip

"""
第五部分：独立部分，视频的处理
"""
read_url = r"E:\python\githubWork\graduation_project\test_videos\challenge.mp4"  # 输入视频文件路径
video_analysis_url = r'E:\python\githubWork\graduation_project\video_analysis'  # 暂存视频帧图片
video_save = r'E:\python\githubWork\graduation_project\new_videos'  # 保存视频的路径


def delete_all(url=video_analysis_url):
    """
    删除文件夹下所有文件
    :param url: 文件目录
    :return: None
    """
    for root, dirs, files in os.walk(url):
        for file in files:
            os.remove(os.path.join(root, file))


def analysis(read_url=read_url, video_analysis_url=video_analysis_url, mod=1):
    """
    用于将视频文件，按帧处理成图片集
    :param read_url: 原视频的路径
    :param video_analysis_url:视频帧的存放路径
    :param mod: 提取的视频的帧数间隔
    :return: None
    """
    cap = cv2.VideoCapture(read_url)
    name_num = 0  # 记录名字的序号
    i = 0

    delete_all()  # 清空中间文件夹下的文件
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        name_num += 1

        name = str(name_num)
        if ret:
            if i % mod == 0:  # 每隔mod帧保存一张图片
                cv2.imwrite(video_analysis_url + '\\' + name + '.jpg', frame)
        else:
            break
    cap.release()


def compound(image_url=video_analysis_url, save_video_url=video_save,fps=24):
    """
    将图像帧转换成视频
    :param image_url: 图像帧的路径
    :param save_video_url: 保存视频的路径
    :param fps: 每秒的帧数
    :return: None
    """
    files = os.listdir(image_url)
    out_num = len(files)
    image_paths = [rf'{image_url}\{i}.jpg' for i in range(1,out_num)]

    clip = ImageSequenceClip(image_paths,fps=fps)   # 读入图片
    clip.write_videofile(save_video_url + '\\' + 'test.mp4')    # 保存视频文件



if __name__ == '__main__':
    compound()
