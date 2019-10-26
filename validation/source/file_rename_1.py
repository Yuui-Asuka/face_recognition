import os

base_path = r'F:\test3'
file_list = os.listdir(base_path)
for i,file in enumerate(file_list):
    file_path = os.path.join(base_path,file)
    new_path = os.path.join(base_path,("person" + ("%04d"%int(i))))
    os.rename(file_path,new_path)