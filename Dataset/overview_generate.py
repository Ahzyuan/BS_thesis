import cv2,sys,os
from glob import glob
import numpy as np

pwd = sys.path[0]
scene_path = os.path.join(pwd, 'pick_snapshot') # pick scene manually
save_path = os.path.join(pwd, 'scene_overview.png') 

scene=glob(scene_path+'\\*')

row=7
column=8

row_storage=[]

for start in range(0,row*column,column):
    row_item=[]
    for idx in range(column):
        row_item.append(cv2.imread(scene[start+idx]))
    row_storage.append(np.hstack(row_item))

res=np.vstack(row_storage)
#res=cv2.resize(res,(1536,756))
cv2.imwrite(save_path,res)
