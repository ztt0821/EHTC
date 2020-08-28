import os
import cv2
from PIL import Image

def unlock_movie(path):

  cap = cv2.VideoCapture(path)
  suc = cap.isOpened()  # 是否成功打开
  frame_count = 0
  while suc:
      frame_count += 1
      suc, frame = cap.read()
      params = []
      params.append(2)  # params.append(1)
      if frame is not None:
          name = '%s\\%dframe.png' %('D:\\dataset\\f_track\\gt_117',frame_count)
          cv2.imwrite(name, frame, params)

  cap.release()
  print('unlock movie: ', frame_count)


# def  jpg_to_video(path, fps):
#   """ 将图片合成视频. path: 视频路径，fps: 帧率 """
#   fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#   images = os.listdir('frames')#os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
#   image = Image.open('frames/' + images[0])
#   vw = cv2.VideoWriter(path, fourcc, fps, image.size)
#
# os.chdir('frames')
# for i in range(len(images)):
#     # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
#     jpgfile = str(i + 1) + '.jpg'
#     try:
#         new_frame = cv2.imread(jpgfile)
#         vw.write(new_frame)
#     except Exception as exc:
#         print(jpgfile, exc)
# vw.release()
# print(path, 'Synthetic success!')


if __name__ == '__main__':
  # PATH_TO_MOVIES = os.path.join('test_movies', 'beautiful_mind2.mp4')
  PATH_TO_MOVIES = 'D:\\dataset\\f_track\\gt_117.flv'
  # PATH_TO_OUTCOME = os.path.join('detection_movies', 'beautiful_mind2_detection_1.avi')
  unlock_movie(PATH_TO_MOVIES)  # 视频转图片
  # jpg_to_video(PATH_TO_OUTCOME, 24)  # 图片转视频