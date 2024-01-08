import os
import cv2

from mian import LaneLine_process

"""
第五部分：独立部分，视频的处理
"""


# 返回所有mp4文件路径
def get_mp4_files(directory=r'test_videos'):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def split_video_into_images(video_path, output_dir='decompose',is_write=False):
    """
    将视频拆分成图片并保存到指定目录中，返回一个包含所有图片的列表。

    参数:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录路径
    返回:
        list: 包含所有图片的列表
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率、帧宽度和帧高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逐帧读取视频并保存为图片，将图片添加到列表中
    frame_count = 0
    image_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_file = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        if is_write:
            cv2.imwrite(output_file, frame)
        image_list.append(frame)
        frame_count += 1

    # 释放视频文件对象
    cap.release()
    return image_list


# 清空decompose下的所有图片
def delete_files_in_directory(directory=r'decompose'):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


# delete_files_in_directory() # 先清空图片
# frames = split_video_into_images(video_paths[1]) # 这里先处理一个视频
# split_video_into_images(video_paths[1]) # 这里先处理一个视频

# 返回decompose下的所有jpg文件路径
def get_jpg_files(directory=r'decompose'):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def ndarray_list_to_video(frames, output_path=r'new_videos/1.mp4'):
    # frames是一个图片列表
    height, width, layers = frames[0].shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # 逐帧将图片写入视频
    for image in frames:
        video.write(image)

    # 释放资源
    cv2.destroyAllWindows()
    video.release()

# output_path = '1.mp4'
# ndarray_list_to_video(frames, output_path)
