import numpy as np
import cv2,os
from sklearn import svm
from HOG_improvement.hog_source_test import Hog
import time,joblib

start=time.time()

def descptor_list(img_list):
    #最终返回对每张图像进行特征提取的列表
    final_desc_list=[]
    counter=0
    #依次加载图像列表中大小为64*64图像：
    for i in img_list:
        counter+=1
        one_vector=[]
        #实例化类
        hog=Hog(i,i.shape[0],i.shape[1])
        descriptor=hog.hog_desc()
        for row in descriptor[:]:
            #切片是列表
            for col in row[:]:
                one_vector.append(col)
        final_desc_list.append(one_vector)
        print('第{}张图片已经形成一维向量'.format(counter))
    return final_desc_list
#改进：这部分可以用joblib做数据然后加载，不用每次加载
if os.path.isfile('img_one_vector_list'):
    print('图形一维向量列表存在，直接加载就行：')
    start5=time.time()
    img_vector_list=joblib.load('img_one_vector_list')
    print('加载图像一维向量列表成功>>>耗时{}'.format(start5-start))
else:
    print('图形一维向量列表不存在，需要调用函数生成')
    start5=time.time()
    #加载64*64 维的图像列表和 label:[[0],[1],[2],[3]......]的标签列表
    data=joblib.load('img_data');
    img_vector_list=descptor_list(data)
    #img_vector_list=np.array(img_vector_list,dtype=np.float32)
    joblib.dump(img_vector_list,'img_one_vector_list')
    print('生成图像一维列表成功>>>耗时{}'.format(start5-start))
label=joblib.load('img_label');
start1=time.time()
print('形成一维向量列表耗时：{}'.format(start1-start))
# 改进：

#2 用形成的图像对应的特征向量列表中的图像向量形成cross_validate：
#for img_vector,label_vector in zip(img_vector_list,label):
def split_set(list_data_set):
    tr_set = []
    te_set = []
    for index, value in enumerate(list_data_set):
        category_num = index // 80
        if index <= category_num*80 + 60:
                tr_set.append(value)
        else:
                te_set.append(value)
    print('交叉验证集完成')
    return tr_set, te_set

tr_data,te_data=split_set(img_vector_list)
tr_label,te_label=split_set(label)
start2=time.time()
print('形成交叉验证集耗时：{}.sec'.format(start2-start1))

clf=svm.SVC(C=1.0,kernel='rbf')
#for img_vector,label_vector in zip(tr_data,tr_label):
    #clf.fit(img_vector,label)

def label2vector(lable2list):
    tr_label1=[]
    for i in lable2list:
        tr_label1.append(i[:])
    vector2v=np.array(tr_label1,dtype=np.int32)
    return vector2v
tr_data=np.array(tr_data,dtype=np.float64)
tr_label=label2vector(tr_label)
clf.fit(tr_data,tr_label)
start3=time.time()
print('SVC 训练耗时：{}.sec'.format(start3-start2))
te_data=np.array(te_data,dtype=np.float64)
te_label=label2vector(te_label)
accuracy=clf.score(te_data,te_label)
print('测试的准确率是:{}%'.format(accuracy*100))
start4=time.time()
print('SVC 测试耗时：{}.sec'.format(start4-start3))
print('------整个程序结束耗时：{}.sec------'.format(time.time()-start))








'''
#不是使用特征提取直接对图像进行分类
def to_one_vector(list_2vector):
    final_scalar=[]
    #按张数进行取图片：i-代表第几章图片
    #如何高效的一次性读入，不用写for循环有待思考
    for i in list_2vector:
        per_img=[]
        img_num=0
        counter=0
        for row in i:
            for val in row:
                per_img.append(val)
                counter+=1
        img_num+=1
        print('{} 张完成'.format(img_num))
        final_scalar.append(per_img)
    return final_scalar
data=to_one_vector(data)
#利用hog.desc()函数进行特征提取
#自制cross_validate
def split_set(data_set):
    tr_set=[];te_set=[]
    for index,value in enumerate(data_set):
        category_num=index//80
        if index<=category_num+60:
            tr_set.append(value)
        else:
            te_set.append(value)
    return tr_set,te_set

train_data,test_data=split_set(data)
train_label,test_label=split_set(label)

clf=svm.SVC(kernel='rbr',C=1.0,)
clf.fit(train_data,train_label)
print(clf.score(test_data,test_label))
'''


