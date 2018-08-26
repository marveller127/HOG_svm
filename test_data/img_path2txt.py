import os
import time
start=time.time()
if os.path.exists('./final.txt'):
# if os.path.isfile('./final.txt')
    os.remove('final.txt')
final_txt=open('final.txt','x')
path=os.path.abspath('test_data')
#test_data_dirs=os.listdir(path)
'''
#方案一：
for dirpath,curdirs,filenames in os.walk(path):
    for dir in curdirs:
        if dir.is_dir:
'''

#方案二：
for dir in os.scandir(path):
    if dir.is_dir():
        label=os.path.basename(dir)
        sub_dir=os.path.abspath(dir)
        sub_dir=os.path.normpath(sub_dir)
        for filename in os.listdir(sub_dir):
            final_txt.write('{}\{};{}\n'.format(sub_dir,filename,label))
final_txt.close()
print('---- the program consumes {}.sec ----'.format(time.time()-start))







