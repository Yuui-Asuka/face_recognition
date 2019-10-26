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

image_size = 160
margin = 44
gpu_memory_fraction = 0.6
class Load:
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.65, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    def __init__(self,model_path,gpu_memory_fraction = 0.6,**kwargs):
        self.image_size = 160
        self.margin = 32
        self.model_path = model_path
        self.gpu_memory_fraction = gpu_memory_fraction

    def load_mtcnn(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return source.detect_face.create_mtcnn(sess, None)

    def load_facenet(self):
        print("loading model......")
        source.facenet.load_model(self.model_path)




    

    

    
