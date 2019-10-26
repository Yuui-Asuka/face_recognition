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
from source.emotion_reco import Emotion
emo = Emotion(train_path_fix = r'source\fix',train_path = r'source\train',test_path = r'source\val',model_path = r'source\model',image_size = 150)
#emo.train_test_clf_model_2()
#emo.train_clf_model()
emo.test_model()