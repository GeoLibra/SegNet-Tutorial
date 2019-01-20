# import os
# import imghdr
# from os.path import join
# from progressbar import ProgressBar
# import shutil
# path='/media/hl/mydata/photos'
# progress=ProgressBar()
# original_images=[]
# unvalid=[]
# for root,dirs,filenames in os.walk(path):
#     print("总图片:",len(filenames))
#     for name in progress(filenames):
#         check=imghdr.what(join(root,name))
#         if not check:
#             unvalid.append(int(name.split('_')[0]))
#             shutil.copy(join(root,name),join('/media/hl/mydata/unvalid',name)) #拷贝文件
#             os.remove(join(root,name))
# unvalid=list(set(unvalid))
# unvalid.sort()
# unvalid=[str(i) for i in unvalid]
# open('unvalid.txt','w+').write('\n'.join(unvalid))


import os
import imghdr
from os.path import join
from progressbar import ProgressBar
from PIL import Image
path='/media/hl/新加卷/mycode/BDStreetView/photos'
progress=ProgressBar()
original_images=[]
unvalid=[]
for root,dirs,filenames in os.walk(path):
    for name in progress(filenames):
        check=join(root,name)
        try:
            img = Image.open(check)
        except IOError:
            unvalid.append(name.split('_')[0])
        try:
            img= np.array(img, dtype=np.float32)
        except:
            unvalid.append(name.split('_')[0])
open('unvalid.txt','w+').write('\n'.join(list(set(unvalid))))
