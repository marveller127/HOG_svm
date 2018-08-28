import cv2,joblib
import numpy as np
# 1.训练集数据部分数据的处理：形成数字化图形列表和标签列表
train_path_file=open('train_img_path.txt','r')
digit_img_list=[];digit_label_list=[]
counter=0
while(1):
    line=train_path_file.readline()
    if line !='':
        splits=line.split(';')
        img_path,label=splits[0],splits[1].strip('\n')
        digit_img=cv2.imread(img_path,0)
        final_img=cv2.resize(digit_img,(32,32))
        counter+=1
        print('训练集中第{}张图片处理完成'.format(counter))
    else:
        break
    digit_img_list.append(final_img)
    digit_label_list.append(np.int64(label))
train_path_file.close()
print('开始序列化 训练 数据>>>')
joblib.dump(digit_img_list,'train_digit_img_list')
joblib.dump(digit_label_list,'train_label_list')
print('训练数据操作完成>>>')
# 2.测试集部分数据的处理：形成数字化图形列表和标签列表
test_path_file=open('test_img_path.txt','r')
digit_img_list=[];digit_label_list=[]
counter=0
while(1):
    line = test_path_file.readline()
    if line !='':
        splits=line.split(';')
        img_path,label=splits[0],splits[1].strip('\n')
        digit_img=cv2.imread(img_path,0)
        final_img=cv2.resize(digit_img,(32,32))
        counter+=1
        print('测试集中的第{}张图片'.format(counter))
    else:
        break
    digit_img_list.append(final_img)
    digit_label_list.append(np.int64(label))
train_path_file.close()
print('开始序列化测试集>>>')
joblib.dump(digit_img_list,'test_digit_img_list')
joblib.dump(digit_label_list,'test_label_list')
print('测试集制作完成')