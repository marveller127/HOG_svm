class Hog:
    import numpy as np
    def __init__(self,img,img_row_num,img_column_num):
        self.__img=img
        self.__row_step=(img_row_num-16)//8+1
        self.__col_step=(img_column_num-16)//8+1

    def hog_desc(self):
        import numpy as np
        import scipy as sp
        '''
        返回值是一个二的数组：
        行数为：Block_row_step*Block_row_step
        列数为：固定的36维
        '''
        #输出多维的步长*步长个Block的36维向量，即vector,length,width;
        final_vector=[]
        counter=0
        thirty_six_vector=[]
        # block_row_step 是block在行方向所能移动的步数[0:self.__row_step]
        # 所涉及的范围是[block_row_step*16,(block_row_step+1)]
        for block_row_step in range(self.__row_step):
            for block_col_step in range(self.__col_step):
                #边界条件：对图像的四个边界进行特殊处理，即舍去
                #if block_col_step==0 and (block_col_step+1)*8==self.__img.shape[0] and block_row_step==0 and (block_col_step+1)*8==self.__img.shape[1]:
                    #continue
                #final_vector=[]
                #对block中的四个cell进行图像操作
                #counter=0

                nine_vector = []
                for cell_row_step in [0,1]:

                    for cell_col_step in [0,1]:
                        #block_left_row
                        #block_right_row
                        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        for row in range(block_row_step*8+cell_row_step*8,block_row_step*8+(cell_row_step+1)*8):
                            for col in range(block_col_step*8+cell_col_step*8,block_col_step*8+(cell_col_step+1)*8):
                                '''
                                this part is testing row and columns :
                                if block_row_step>1 or block_col_step>1:
                                    break
                                else:
                                    print(row,col)
                                    counter+=1
                                    '''
                                counter+=1
                                if row == 0 or row+1== self.__img.shape[
                                    0] or col==0 or col+1==self.__img.shape[1]:
                                    # 打印边界条件的点
                                    #print('conditional_bouding_point:{}'.format((row,col)))
                                    continue
                                #像素为ubyte类型[0-255],相减得到负数会溢出，应该讲负数转成int
                                x_devate=int(self.__img[row,col+1])-int(self.__img[row,col-1])
                                y_devate=int(self.__img[row+1,col])-int(self.__img[row-1,col])
                                if y_devate==0:
                                    thea=90
                                else:
                                    thea=np.arctan(x_devate/y_devate)*180/(np.pi)
                                    if thea<0:
                                        thea=-thea
                                slide_len=sp.sqrt(x_devate**2+y_devate**2)
                                vector_index=int(thea//20)
                                # 按列形成 9 维向量
                                vector[vector_index]=vector[vector_index]+slide_len
                        # 按行每个cell形成 9维向量,共有4个cell
                        nine_vector.append(vector)
                        # 将形成的 9 维向量放入 2*2 的容器中形成4个9维
                nine_vector_np=np.array(nine_vector).reshape(1,36)
                #将每个Block中的36维向量变成一个一维向量
                thirty_six_vector.append(nine_vector_np)
                #36维的这个列表里边放的是1*36 的向量
        #thirty_six_vector_np=(np.array(thirty_six_vector)).reshape(self.__row_step,self.__col_step,36)
        thirty_six_vector_np=np.array(thirty_six_vector)
                #将移动的步数形成多个36维向量放到一起
        #final_vector.append(thirty_six_vector_np)
        #final=np.array(final_vector)
        descriptor=thirty_six_vector_np.reshape((self.__row_step*self.__col_step),36)
        return descriptor

    def __str__(self):
        return '数组为[Block_row_num*Block_col_num,36]'

def count_time():
    #这是个实现计时功能的装饰函数
    pass

if __name__=='__main__':
    import time
    import numpy as np
    import scipy as sp
    import cv2
    start=time.time()
    input_grag_pict=cv2.imread('23.jpg',0)
    norm_img=cv2.resize(input_grag_pict,dsize=(64,64))
    cv2.imshow('tst',norm_img)
    #cv2.waitKey()
    img_row,img_col=norm_img.shape[0],norm_img.shape[1]
    hog=Hog(norm_img,img_row,img_col)
    descriptor=hog.hog_desc()
    print(descriptor.shape)
    #len(descriptor)
    '''print('the program has consumes {}.sec'.format(time.time()-start))
    print(descriptor.shape)
    print(descriptor.reshape(length*width,36).shape)
    #print('the descriptor has {} vector'.format(len(descriptor)))

    #final_desc=np.array(len(descriptor),9)
'''



































