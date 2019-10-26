import os
import sys
"""该程序用于文件的重命名，请先将每个人的照片存放在各自的文件夹中，
将文件夹改为人名，之后将会为图片进行重命名
"""
base_path = r'E:\lfw_160'
file_list = os.listdir(base_path)
files = []
for file in file_list:
    path = os.path.join(base_path,file)
    files.append(path)
count2 = 0
for file in files:
    photo_list = os.listdir(file)
    count = 1
    for photo in photo_list:
        photo_path = os.path.join(file,photo)
        #people_name = os.path.splitext(photo)[0].split()[0]
        people_name = file.split(os.sep)[-1]
        file_type = os.path.splitext(photo)[1]
        new_name = people_name + " " + "(" +str(count) + ")" + file_type
        count+=1
        count2+=1
        new_path = os.path.join(file,new_name)
        print(new_path)
        os.rename(photo_path,new_path)
        print("rename %s files" %count2)
        

