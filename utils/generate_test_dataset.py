# -*- coding: utf-8 -*-
import os,shutil
import numpy as np

def generate():
   
    fileList = os.listdir(r"/data")
    
    file_index = [i for i in range(0,len(fileList))]
    np.random.shuffle(file_index)
    choose_index = file_index[0:200000]
    fileList = np.array(fileList)
  
    test_sample = fileList[choose_index]
   
    i = 0
    for person_id in test_sample:
        
        id_path = "/data/"+person_id+'/'
        output_path = "/data/"
        
        fileList = os.listdir(id_path)
        for person_picture in fileList:
            if not os.path.exists(output_path):
                os.makedirs(output_path)  # 创建路径
            shutil.move(id_path+person_picture, output_path)  # 移动文件
            i+=1
            if i>=200000:
                return 
            print "move %s -> %s" % (id_path+person_picture, output_path)
 

def modify():

    fileList = os.listdir(r"/data/")
    for person_id in fileList:
        id_path = "/data/" + person_id + '/'
        fileList = os.listdir(id_path)
        i = 0
        for person_picture in fileList:
            i+=1
            ori_name = id_path + person_picture
            new_name = id_path + person_id+ "_" + "%04d.png"%(i)
            os.rename(ori_name, new_name)  # 重命名文件
            print "rename %s -> %s" % (ori_name,new_name)            

if __name__ == '__main__':
    generate()
    #modify()
