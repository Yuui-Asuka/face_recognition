import cv2
import time
import sys
import os
import uuid
import numpy as np
import source.facenet
import tensorflow as tf
from PIL import Image,ImageDraw,ImageFont
import queue
import matplotlib.pyplot as plt
from source.recognise_v2 import Recognition
from source.emotion_reco import Emotion
from source.store_face import WriteEmb
from source.load_model import Load



class PictureHandle:
    def add_frame_en(self,cv_img,name_list,bounding_boxes,emotion_list):
        size = cv_img.shape[0]//400
        size_2 = cv_img.shape[0]//100
        cv_img_ = cv_img.copy()
        if bounding_boxes is not None:
            for i,bb in enumerate(bounding_boxes):
                face_bb = bb.astype(int) 
            #显示边框
                cv2.rectangle(cv_img_,(face_bb[0],face_bb[1]),(face_bb[2],face_bb[3]),
                              (0,255,0),size_2)
            #显示名字
                if name_list is not None:
                    cv2.putText(cv_img_,name_list[i],(face_bb[0],face_bb[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,size,(0,255,0),
                                thickness = size, lineType = 2)
                    cv2.putText(cv_img_,emotion_list[i],(face_bb[0],face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX,size,(0,255,0),
                                thickness = size,lineType = 2)
        return np.array(cv_img_)

    def add_frame_ch(self,cv_img,name_list,bounding_boxes,emotion_list):
        size = cv_img.shape[0]//12
        size_2 = cv_img.shape[0]//100
        chinese_dict = {'happy':'高兴','unhappy':'不高兴','surprise':'惊讶','angry':'愤怒'}
        cv_img_ = cv_img.copy()
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
                    draw.text((face_bb[0],face_bb[2]),chinese_dict[emotion_list[i]],(255,0,0),font = font)
        return np.array(cv_img_)
    
    @classmethod
    def add_mark(cls,file_path,save_path,use_chinese = False,use_multi = False):
        """name_list:when more than one face in a picture, it will return a name list:
	   [person1,person2,person3...];
	   bounding_boxes:four coordinate about a face. when more than one face in a picture,
	   it will be a list: [coord1,coord2,coord3...]
        """
        if not isinstance(file_path,(str,tuple,list)):
            raise TypeError("this path can be a string,tuple or list")
        #rec = Recgnition()
        i = 0
        namespace = uuid.NAMESPACE_URL
        q = queue.Queue()
        while q.empty():
            q.put(file_path)
        while not q.empty():
            image_path = q.get()
            i += 1
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                try:
                    cv_img = np.array(Image.open(file_path))
                    if cv_img.ndim == 2:
                        cv_img = source.facenet.to_rgb(cv_img) 
                        cv_img = cv_img[:,:,0:3]
                    elif cv_img.ndim == 3 and cv_img.shape[2] > 3:
                        cv_img = cv_img[:,:,0:3]
                except IOError as e:
                    print('error',e)
            else:
                print("this picture can not be read.")
            name_list,bounding_boxes = rec.get_name_and_box(cv_img,p_net,r_net,o_net,sess)
            images,_ = rec.find_faces(cv_img,p_net,r_net,o_net)
			#使用一对一模型
            if use_multi == False:
                emotion_list = emo.image_recognise(images,use_multi = False)
			#使用一对多模型
            else:
                emotion_list = emo.image_recognise(images,use_multi = True)
            if use_chinese == False:
                cv_img = cls().add_frame_en(cv_img,name_list,bounding_boxes,emotion_list)
            else:
                cv_img = cls().add_frame_ch(cv_img,name_list,bounding_boxes,emotion_list)
           
            pic_path = os.path.join(save_path,("%04d"%int(i)+'.jpg'))
            pic_uuid = str(uuid.uuid3(namespace = namespace,name = ("%04d"%int(i)))) + ".jpg"
            plt.imsave(pic_path,cv_img)
            yield [cv_img,pic_path,pic_uuid]

class Mark:
    """
    mark = Mark(file_path= [path1,path2,path3....] 或者 "path" 或者 (path1,path2,path3...),save_path = "path"
            use_chinese = False)
    example:
    for i in range(5):
        image_info = mark.mark()
    image_info: generater,if you put more than one picture in a list or a tuple,use loop to get the information
    image_info:list[image,pic_path,pic_uuid]

    """
    def __init__(self,file_path,save_path,use_chinese = False,use_multi = False):
        self.pic = PictureHandle()
        self.file_path = file_path
        self.save_path = save_path
        self.use_chinese = use_chinese
        self.use_multi = use_multi


    def mark(self):
        image_info = self.pic.add_mark(file_path=self.file_path,save_path=self.save_path,
                                                    use_chinese=self.use_chinese,use_multi = self.use_multi)
        for i in image_info:
            yield i

load = Load(model_path = r'source\20180402-114759')
Wb = WriteEmb(r"source\data\multi",reinforce = False,noise = True,embedding_size = 128)
emo = Emotion(train_path_fix = r'source\fix',train_path = r'source\train',test_path = r'source\val',model_path = r'source\model',image_size = 150)
with tf.Session() as sess:
	p_net,r_net,o_net = load.load_mtcnn()
	load.load_facenet()
	Wb.get_emb_and_write(sess,p_net,r_net,o_net)
	rec = Recognition()
	mark = Mark(file_path = r'F:\camera2\001.jpg',save_path = '.\source',use_chinese = True,use_multi = False)
	image_info = mark.mark()
	for i in image_info:
		plt.imshow(i[0])
		plt.show()