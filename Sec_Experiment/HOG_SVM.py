import numpy as np
import cv2,os
from sklearn import svm
from HOG_minist.hog_source_test  import Hog
import time,joblib

start=time.time()

def descptor_list(img_list,label_file,counter_num):
    #最终返回对每张图像进行特征提取的列表
    final_desc_list=[]
    counter=0
    label_list=[]
    label=joblib.load('{}'.format(label_file))
    #依次加载图像列表中大小为64*64图像：
    for i in img_list:
        if counter == counter_num:
            break
        one_vector=[]
        #实例化类
        hog=Hog(i,i.shape[0],i.shape[1])
        descriptor=hog.hog_desc()
        for row in descriptor[:]:
            #切片是列表
            for col in row[:]:
                one_vector.append(col)
        print('第{}张图片已经形成一维向量'.format(counter))
        final_desc_list.append(one_vector)
        print('开始加载{}标签'.format(counter))
        label_list.append(label[counter])
        counter+=1

    return final_desc_list,label_list
#改进：这部分可以用joblib做数据然后加载，不用每次加载
if os.path.isfile('train_img_one_vector_list') and os.path.isfile('test_img_one_vector_list'):
    print('图形一维向量列表存在，直接加载就行：')
    start5=time.time()
    train_img_vector_list=joblib.load('train_img_one_vector_list')
    tr_label=joblib.load('tr_label')
    test_img_vector_list=joblib.load('test_img_one_vector_list')
    te_label=joblib.load('te_label')
    print('加载图像一维向量列表成功>>>耗时{}'.format(start5-start))
else:
    print('图形一维向量列表不存在，需要调用函数生成')
    start5=time.time()
    #加载64*64 维的图像列表和 label:[[0],[1],[2],[3]......]的标签列表
    print('开始生成测试集:')
    test_data=joblib.load('test_digit_img_list')
    test_img_vector_list,te_label=descptor_list(test_data,'test_label_list',1000)
    joblib.dump(test_img_vector_list, 'test_img_one_vector_list')
    joblib.dump(te_label,'te_label')
    #del test_img_vector_list
    print('测试集特征生成完毕！！！')
    print('开始生成训练集:')
    train_data = joblib.load('train_digit_img_list')
    train_img_vector_list,tr_label= descptor_list(train_data,'train_label_list',6000)
    joblib.dump(train_img_vector_list, 'train_img_one_vector_list')
    joblib.dump(tr_label,'tr_label')
    #del train_img_vector_list
    print('训练集特征生成完毕！！！')
    #img_vector_list=np.array(img_vector_list,dtype=np.float32)
    print('生成图像一维列表成功>>>耗时{}'.format(start5-start))

start1=time.time()
print('形成一维向量列表耗时：{}'.format(start1-start))
# 改进：

#2 用形成的图像对应的特征向量列表中的图像向量形成cross_validate：

start2=time.time()
print('形成交叉验证集耗时：{}.sec'.format(start2-start1))
print('开始训练')
clf=svm.SVC(C=1.0,kernel='rbf')
#for img_vector,label_vector in zip(tr_data,tr_label):
    #clf.fit(img_vector,label)
'''
def label2vector(lable2list):
    tr_label1=[]
    for i in lable2list:
        tr_label1.append(i[:])
    vector2v=np.array(tr_label1,dtype=np.int32)
    return vector2v
'''
tr_data=np.array(train_img_vector_list,dtype=np.float64)
#tr_label=label2vector(tr_label)
tr_label=np.array(tr_label,dtype=np.int32)
clf.fit(tr_data,tr_label)
print('训练结束>>>')
start3=time.time()
print('SVC 训练耗时：{}.sec'.format(start3-start2))
print('开始测试:')
te_data=np.array(test_img_vector_list,dtype=np.float64)
#te_label=label2vector(te_label)
te_label=np.array(te_label,np.int32)
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


