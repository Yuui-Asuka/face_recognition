import cv2
import time
import sys
import source.facenet
import freetype
from PIL import Image,ImageDraw,ImageFont
from source.video_recognize import Recognition
import numpy as np
frame_interval = 3
fps_display_interval = 5
frame_rate = 0
frame_count = 0

def add_frame(cv_img,frame_rate):
    if bounding_boxes is not None:
        for i,bb in enumerate(bounding_boxes):
            face_bb = bb.astype(int) 
            #显示边框
            cv2.rectangle(cv_img,(face_bb[0],face_bb[1]),(face_bb[2],face_bb[3]),
                          (0,255,0),2)
            #显示名字
            if name_list is not None:
                cv2.putText(cv_img,name_list[i],(face_bb[0],face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),
                            thickness = 2, lineType = 2)
    #显示FPS
    cv2.putText(cv_img, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def add_frame_ch(cv_img,name_list,bounding_boxes,frame_rate):
    size = cv_img.shape[0]//12
    size_2 = cv_img.shape[0]//100
    chinese_dict = {'happy':'高兴','unhappy':'不高兴','surprise':'惊讶','angry':'愤怒'}
    cv_img_ = cv_img.copy()
    cv2.putText(cv_img_, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    if bounding_boxes is not None:
        for i,bb in enumerate(bounding_boxes):
            face_bb = bb.astype(int) 
            #显示边框
            cv2.rectangle(cv_img_,(face_bb[0],face_bb[1]),(face_bb[2],face_bb[3]),
                              (0,255,0),size_2)
            if name_list is not None:
                cv_img_ = Image.fromarray(cv_img_.astype('uint8'))
                draw = ImageDraw.Draw(cv_img_)
                font = ImageFont.truetype("source\simhei.ttf",size = size,encoding = 'utf-8')
                draw.text((face_bb[0],face_bb[1]),name_list[i],(255,0,0),font = font)
              #  draw.text((face_bb[0],face_bb[3]),chinese_dict[emotion_list[i]],(255,0,0),font = font)
    return np.array(cv_img_)

if __name__ == "__main__":
    #加载视频
    #cap = cv2.VideoCapture('source/aaa.mp4')
    #加载摄像头
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    rec = Recognition()

    while cap.isOpened():
        ret, cv_img = cap.read()
        name_list,bounding_boxes = rec.get_name_and_box(cv_img)
        if cv_img is None:
            break
        if (frame_count % frame_interval) == 0:
        #计算FPS
            end_time = time.time()
            times = end_time - start_time
            if times > fps_display_interval:
                frame_rate = int(frame_count / times)
                start_time = time.time()
                frame_count = 0
    #标注中文
        cv_img = add_frame_ch(cv_img,name_list,bounding_boxes,frame_rate)
    #标注英文
    #add_frame(cv_img,frame_rate)
        frame_count += 1
        cv2.imshow('Video',cv_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
