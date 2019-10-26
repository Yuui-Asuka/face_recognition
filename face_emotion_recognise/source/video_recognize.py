from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import source.facenet
import source.detect_face

model_path = r'source/facemodel_002'
image_size = 160
margin = 44
gpu_memory_fraction = 0.6

class Library:
    def __init__(self):
        self.emb_path = r'source\emb.txt'

    def get_face_library(self):
        names = []
        arrays = []
        with open(self.emb_path,'r') as fp:
            for data in fp.readlines():
                names.append(data.split(':')[0])
                arrays.append(eval(data.split(':')[1]))
        return names,arrays

class faceDetect:
    minsize = 20  # minimum size of face
    threshold = [0.8, 0.85, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor
    def __init__(self):
        self.image_size = image_size
        self.margin = margin
        self.gpu_memory_fraction = gpu_memory_fraction     
        self.pnet,self.rnet,self.onet = self.load_mtcnn()
      
    def load_mtcnn(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return source.detect_face.create_mtcnn(sess, None)
       
    def find_faces(self,image):
        img_size = np.asarray(image.shape)[0:2]
        bounding_boxes, _ = source.detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        faces = []
        if bounding_boxes.shape[0] > 0:
            for bb in bounding_boxes:
                bounding_box = np.zeros(4, dtype=np.int32)
                img_size = np.asarray(image.shape)[0:2]
                bounding_box[0] = np.maximum(bb[0] - self.margin / 2, 0)
                bounding_box[1] = np.maximum(bb[1] - self.margin / 2, 0)
                bounding_box[2] = np.minimum(bb[2] + self.margin / 2, img_size[1])
                bounding_box[3] = np.minimum(bb[3] + self.margin / 2, img_size[0])
                cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
                aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                prewhitened = source.facenet.prewhiten(aligned)
                faces.append(prewhitened)
            images = np.stack(faces)
        else:
            images = []
        return images,bounding_boxes

class Recognition:     
    def __init__(self):              
        library = Library()
        self.face_det = faceDetect()
        self.names,self.arrays = library.get_face_library()
        self.model_path = model_path
        self.sess = tf.Session()      
        with self.sess.as_default():
            source.facenet.load_model(self.model_path)
        
    def get_name_and_box(self,img):
        names = self.names
        image,bounding_boxes = self.face_det.find_faces(img)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        if len(image) > 0:
            feed_dict = { images_placeholder: image, phase_train_placeholder:False }
            emb = self.sess.run(embeddings, feed_dict=feed_dict)
        else:
            emb = []
        name_list = []
        img_array = np.array(self.arrays)
        nrof_people = img_array.shape[0]
        if len(emb) > 0:
            nrof_faces = emb.shape[0]
            distance = np.zeros(shape = (nrof_faces,nrof_people))
            for i in range(nrof_faces):
                for j in range(nrof_people):
                    dist = np.mean([(np.sqrt(np.sum(np.square(np.subtract(emb[i,:], img_array[j,k,:]))))) for k in range(5)]) 
                    if dist > 1.16:
                        dist = np.nan
                    distance[i][j] = dist
            sorted_dis = np.argsort(distance,axis = 1)
            for i in range(nrof_faces):
                top1_dis = sorted_dis[i][0]
                name = names[top1_dis]
                if (np.isnan(distance[i][:])).tolist() == [True for n in range(nrof_people)]:
                    name = "unknown"
                name_list.append(name)
        return name_list,bounding_boxes
