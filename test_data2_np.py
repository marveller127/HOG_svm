import cv2
import numpy as np
import joblib
file1=open('final.txt','r')
data=[];label=[]
counter=0
while(1):
    line=file1.readline()
    #line.strip('\n')
    if len(line)!=0:
        line=line.split(';')
        path_str,label_str=line[0],line[1].strip('\n')
        img=cv2.imread(path_str,0)
        pict=cv2.resize(img,(64,64))
        data.append(pict)
        label.append(label_str)
        counter+=1
        print('第%d行完成'%counter)
    else:
        break
file1.close()
# 形成64*64的图像列表
joblib.dump(data,'img_data')
joblib.dump(label,'img_label')



