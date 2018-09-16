#coding=UTF-8
import numpy as np
import cv2
import sys
from PIL import Image
from align_dlib import AlignDlib 
import os

def process(person_id, video_path,output_path):

    ad = AlignDlib("/data/shape_predictor_68_face_landmarks.dat")
    catch_pic_num = 5
    cap = cv2.VideoCapture(video_path)
    color = (0, 255, 0)
    num = 0
    try_num = 0 
    while cap.isOpened():
        print("进入检测")
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            try_num += 1
            print("未读取到")
            break
        if try_num > 100:
            print("连续失败100次，退出")
            break;

        size = frame.shape[0]
        output_path = output_path + '%s_0000.jpg'%(person_id)    
        if(ad.align(size,frame) is not None):
            cv2.imwrite(output_path,frame)
            break;

        frame90 = np.rot90(frame)
        if(ad.align(size,frame90) is not None):  
            cv2.imwrite(output_path,frame90)
            break;
                   
        frame180 = np.rot90(np.rot90(frame))
        if(ad.align(size,frame180) is not None):
            cv2.imwrite(output_path,frame180)
            break;
            
        frame270 = np.rot90(np.rot90(np.rot90(frame)))
        if(ad.align(size,frame270) is not None):
            cv2.imwrite(output_path,frame270)
            break;
           
        print("未提取成功，等待下一帧")
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
        cap.release()

if __name__ == '__main__':

    path_name="./videos"
    fileList = os.listdir(path_name)

    for person_id in fileList:
        id_path = path_name+'/'+person_id+'/'
        output_path = './generate/'+person_id+'/'
        fileList = os.listdir(id_path)
        for person_file in fileList:
            if not os.path.exists(output_path):
                os.makedirs(output_path)  # 创建路径
            if person_file.endswith(".mp4"):
                process(person_id,id_path+person_file,output_path)
