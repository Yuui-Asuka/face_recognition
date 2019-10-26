from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import copy
import sys
import os
import source.facenet
import source.detect_face
import matplotlib.pyplot as plt
import random
import numpy as np
import skimage
from PIL import Image

class Generate(object):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.65, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    def __init__(self,image_path,**kwargs):
        self.image_size = 160
        self.margin = 32
        self.image_path = image_path

    def image_reinforce(self):
        image_path = self.image_path
        people_paths = []
        for file in os.listdir(image_path):
            people_path = os.path.join(image_path,file)
            people_paths.append(people_path)

        reinforce_names = []
        for file in people_paths:
            image_list =  os.listdir(file)
            if len(image_list) <= 2:
                reinforce_names.append(file)

        for name in reinforce_names:
            images_list = [os.path.join(name,img) for img in os.listdir(name)]
            random_reinforce_pic = random.choice(images_list)
            image_raw_data = tf.gfile.FastGFile(random_reinforce_pic,'rb').read()
            img_data = tf.image.decode_image(image_raw_data)
            img_data = tf.image.convert_image_dtype(img_data,dtype = tf.float32)
            evaled = img_data.eval()
            dark = tf.image.adjust_brightness(img_data,-52./255.)
            bright = tf.image.adjust_brightness(img_data,52./255.)
            ga_noise = skimage.util.random_noise(evaled,mode = 'gaussian',
                                          seed = None,clip = True,
                                          var = 0.02)
            salt_noise = skimage.util.random_noise(evaled,mode = 'salt',seed = None,clip = True)
            pepper_noise = skimage.util.random_noise(evaled,mode = 'pepper')
            shelt = evaled.copy()
            a,b,c = evaled.shape
            x_coord = random.randint(a//10,a*9//10)
            y_coord = random.randint(b//10,b*9//10)
            for i in range(a):
                for j in range(b):
                    center = np.array([x_coord,y_coord])
                    t = np.array([i,j])
                    if np.sqrt(np.sum(np.square(t-center))) < a/15:
                        shelt[i,j,1] = 38/255
                        shelt[i,j,2] = 42/255
                        shelt[i,j,0] = 67/255
            dark = tf.image.encode_jpeg(tf.image.convert_image_dtype(dark,dtype = tf.uint8))
            bright = tf.image.encode_jpeg(tf.image.convert_image_dtype(bright,dtype = tf.uint8))
            salt_noise = tf.image.encode_jpeg(tf.image.convert_image_dtype(salt_noise,dtype = tf.uint8))
            pepper_noise = tf.image.encode_jpeg(tf.image.convert_image_dtype(pepper_noise,dtype = tf.uint8))
            shelt = tf.image.encode_jpeg(tf.image.convert_image_dtype(shelt,dtype = tf.uint8))
            for i,reinforced in enumerate([dark,bright,salt_noise,pepper_noise,shelt]):
                with tf.gfile.GFile(os.path.join(name,(str(i)+'.jpg')),'wb') as f:
                    f.write(reinforced.eval())

    def find_faces(self,image_path,p_net,r_net,o_net):
        old_name_list = []
        new_name_list = os.listdir(image_path)
        file_list = []
        open(r'source\emb.txt','a')
        with open(r'source\emb.txt','r') as fp:
            for data in fp.readlines():
                old_name_list.append(data.split(':')[0])
        add_name = list(set(old_name_list)^set(new_name_list))
        for file in add_name:
            filepath = os.path.join(image_path,file)
            file_list.append(filepath)
        people_images = []
        for i,file in enumerate(file_list):
            images_file = []
            images_file.append(os.listdir(file))
            random_image = np.random.randint(0,len(images_file[0]), size = 5)
            people_images_name = []
            people_image_path = []
            for x,im in enumerate(random_image):
                people_images_name.append(images_file[0][im])
                people_image_path.append(os.path.join(file_list[i],people_images_name[x]))
            people_images.append(people_image_path)
        tmp_image_paths = copy.copy(people_images)
        imim = []
        for people in tmp_image_paths:
            img_list = []
            for image in people:
                img = misc.imread(os.path.expanduser(image),mode = 'RGB')
                img_size = np.asarray(img.shape)[0:2]
                bounding_boxes, _ = source.detect_face.detect_face(img, self.minsize, pnet = p_net, rnet = r_net, onet = o_net,
                                                           threshold = self.threshold, factor = self.factor)
                assert len(bounding_boxes) >= 1, "this picture dees not have face!"
                det = np.squeeze(bounding_boxes[0,0:4])

                bounding_box = np.zeros(4, dtype=np.int32)
                img_size = np.asarray(img.shape)[0:2]
                bounding_box[0] = np.maximum(det[0] - self.margin / 2, 0)
                bounding_box[1] = np.maximum(det[1] - self.margin / 2, 0)
                bounding_box[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                bounding_box[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                cropped = img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
                aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                prewhitened = source.facenet.prewhiten(aligned)
                img_list.append(prewhitened)
                images = np.stack(img_list)
            imim.append(images)
            
        return imim,add_name

    def get_emb(self,p_net,r_net,o_net,sess,reinforce = False,noise = False,embedding_size = 128):
        if reinforce == True:
            self.image_reinforce()

        images,name_list = self.find_faces(image_path = self.image_path,p_net = p_net,r_net = r_net,o_net = o_net)
        embs = []
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        for image in images:
            feed_dict = { images_placeholder:image,phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict = feed_dict)
            if noise == True:
                for e in emb:
                    noise = np.random.normal(0,0.02,embedding_size)
                    emb = e + noise
            embs.append(emb)

        name_to_emb = list(zip(name_list,embs))
        with open(r'source\emb.txt','a') as fp:
            for i in range(len(name_to_emb)):
                s1 = str(name_to_emb[i][0]) + ':'
                s2 = ''.join(str(name_to_emb[i][1].tolist())) + '\n'
                s = s1+s2
                fp.write(s)

class WriteEmb(Generate):
    def __init__(self,image_path,**kwargs):
        super().__init__(image_path,**kwargs)
        self.reinforce = kwargs['reinforce']

    def get_emb_and_write(self,sess,p_net,r_net,o_net):
        self.get_emb(p_net,r_net,o_net,sess,self.reinforce)

