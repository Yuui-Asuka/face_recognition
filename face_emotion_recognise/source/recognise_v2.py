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

class Recognition:
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.6, 0.7]  # three steps's threshold
    factor = 0.709 #scale factor

    def __init__(self):
        library = Library()
        self.names,self.arrays = library.get_face_library()
        self.margin = 44
        self.image_size = 160

    def find_faces(self,image,p,r,o):
        img_size = np.asarray(image.shape)[0:2]
        bounding_boxes, _ = source.detect_face.detect_face(image, self.minsize, pnet = p, rnet = r, onet = o,threshold = self.threshold, factor = self.factor)
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

    def get_name_and_box(self,img,p_net,r_net,o_net,sess):
        names = self.names
        image,bounding_boxes = self.find_faces(img, p_net, r_net, o_net)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        if len(image) > 0:
            feed_dict = { images_placeholder: image, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
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
                    if dist > 1.05:
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