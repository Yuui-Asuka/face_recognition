import os 
import numpy as np
from PIL import Image
"""该脚本用于确保训练集中不会出现异常文件"""
file_path = r"train_set2"
images = []
for root,dirs,files in os .walk(file_path):
    for file in files:
        image_path = os.path.join(root,file)
        images.append(image_path)
        
for p in images:
    x = p.split(os.sep)[-1]
    y = os.path.splitext(x)[-1]
    if y != ".png" or y != ".jpg":
        print(p)
    img = np.array(Image.open(p))
	if file_path == "train_set2":
		if img.shape != (224,224,3):
			print(p)
	elif file_path == "train_set":
		if img.shape != (160,160,3):
			print(p)
