from multiprocessing import Process
import os
import imghdr
from os.path import join
from progressbar import ProgressBar
import shutil
path='/media/hl/新加卷/mycode/BDStreetView/photos'

def process_data(filelist):
    unvalid=[]
    for filepath in filelist:
        filename = os.path.basename(filepath)
        check=imghdr.what(filepath)
        if not check:
            fpath = os.path.dirname(filepath)
            name=filename.split('_')[0]
            unvalidpath=fpath.split('/')[:-1]
            unvalid.append(name)
            shutil.copy(filepath,join('/'.join(unvalidpath)+'/unvalid',filename)) #拷贝文件
            os.remove(filepath)
    open('unvalid2.txt','a').write('\n'.join(unvalid))
if __name__=="__main__":
    full_list=[]
    for root,dirs,filenames in os.walk(path):
        for name in filenames:
            full_list.append(join(root,name))
    n_total=len(full_list)
    n_processes=32
    length=n_total/n_processes
    indices=[int(round(i*length)) for i in range(n_processes+1)]
    # 生成每个进程要处理的子文件列表
    sublists=[full_list[indices[i]:indices[i+1]] for i in range(n_processes)]
    # 生成进程
    processes=[Process(target=process_data,args=(x,)) for x in sublists]
    # 并行处理
    for p in processes:
        p.start()
    for p in processes:
        p.join()