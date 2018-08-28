* # 第二实验
## 1. 从[Mnist官网](http://yann.lecun.com/exdb/mnist/)下载下图蓝框中的四个数据集
## ![如图所示](https://github.com/marveller127/HOG_svm/blob/master/Sec_Experiment/Mnist.jpg)
## 2. 将下载的四个文件减压，放到这次实验文件夹下的名为“ Mnist ”的文件夹下，在“ Mnist ”同级的文件夹下创建“ train ”和“ test ”两个文件夹，分别用来盛放训练图片与标签文件和测试图片与标签文件；<br/>![如下图所示](https://github.com/marveller127/HOG_svm/blob/master/Sec_Experiment/%E7%BB%93%E6%9E%84.jpg)
## 3. 分别运行‘ train_ubyte2img.py ’和“ test_ubyte2img_mod.py ”将对应的图片和对应的图片类别的文件加载到指定的路径下；运行完毕之后在 train 文件和 test 文件夹下分别找到 train_label.txt 和 test_label.txt 文件，将其放到上一级目录中（即和 train 和 test 同一级的目录下）
## 4. 分别运行 train_img_path_and_label.py 和test_img_path_and_label.py 文件，在 Mnist 同级文件夹下分别生成：<br/> (1)  train_img_path.txt 文件（将训练图片的路径和对应的标签存成txt文件方便数据的训练）和<br/> (2) test_img_path.txt文件（将测试图片的路径和对应的标签存成txt文件方便数据的测试）
## 5. 运行 genernate32_32img_data_label 依次将路径中的训练与测试用的数字图像标准化成32的尺寸放入一个列表中，并将列表序列化；依次将图像对应的标签放入一个列表中，并将列表序列化；运行完毕生成 train_digit_img_list，train_label_list ；test_digit_img_list， test_label_list 四个文件
## 6. 使用自己写的源码 hog_source_test.py 中hog类，该类中有实现提取Block-36维向量的描述子的hog_desc()的方法；
## 7. 最后，运行 HOG_SVM.py 脚本，对每张图像进行特征提取，形成一个Block_row_num乘以Block_col_num乘以36的一维向量，将每一张图像形成的特征插入到一个列表中，使用numpy将这个列表转成一个二维数组，每一行都是一张图像的特征，加载图片对应的标签使用 klearn 中的 SVM 进行训练，使用测试数据集进行测试，计算 accuracy
