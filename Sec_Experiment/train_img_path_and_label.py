import os,time

start=time.time()
if os.path.exists('train_img_path.txt'):
    print('图像路径文件已经存在>>>')
else:
    print('图像路径文件不存在>>>')
    train_label=open('train_label.txt','r')
    all_label=train_label.readlines()
    label_list=[]
    for label_index,label_val in enumerate(all_label[0]):
        if label_index%2==0:
            label_list.append(label_val)

    path_file=open('train_img_path.txt','x')
    img_list=os.listdir('train')
    abs_path=os.path.abspath('train')
    for path1,label1 in enumerate(label_list):
        path_file.write('{}\n'.format(abs_path+'\\'+str(path1)+'.png'+';'+label1))
        print('第{}张图片写入成功：'.format(path1))
    print('----- 程序耗时{}.sec -----'.format(time.time()-start))
    path_file.close()
