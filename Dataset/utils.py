import os,shutil
import numpy as np
from tqdm import tqdm
from glob import glob

def gap_pick(origin_path, target_num=12000):
    '''
    Automatically pick gap according to the number of origin images and target images 
    Then, pick with generated gap from origin images, copy the pick images to `pick_img` file.
    '''
    assert os.path.exists(origin_path), f'pictures\' original path {origin_path} doesn\'t exist'
    new_path=os.path.split(origin_path)[0]+'/pick_img'
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)

    img_list=os.listdir(origin_path)[::-1]
    origin_num=len(img_list)
    
    # Calculate the gap between images
    gap=round((origin_num/(target_num-1))-1)
    if gap<1:
        gap=1

    # Generate the indices of the images to be picked
    idx_generate=(gap+1)*(np.arange(1,origin_num+1)-1)  #(gap+1)*(np.arange(1,target_num+1)-1)
    # Make sure the indices are within the range of the number of images
    idx_generate=idx_generate[idx_generate<=origin_num]

    img_list=np.array(img_list)
    pick_list=img_list[idx_generate]

    # Copy the picked images to the new path
    for img in tqdm(pick_list):
        img_origin_path=os.path.join(origin_path,img)
        img_new_path=os.path.join(new_path,img)
        shutil.copyfile(img_origin_path,img_new_path)

def match_pick_label(img_path,label_path):
    '''
    delete the label file that not match the image file
    '''
    img_list='\n'.join(os.listdir(img_path)).replace('.png','.txt')
    img_list=set(img_list.split('\n'))
    label_list=set(os.listdir(label_path))

    match_label=label_list-img_list

    for label in tqdm(match_label):
        os.remove(os.path.join(label_path,label))

def cal_instance_num(label_path):
    '''
    Calculate the number of labeled instances of each class
    '''
    label_list=glob(label_path+'/*')
    cls_num=[0,0,0] # num_T, num_Z, num_P

    def cal_bycls(line):
        cls_idx=int(line.strip()[0])
        cls_num[cls_idx]+=1

    for label in label_list:
        with open(label,'r',encoding='utf-8') as label_reader:
            ins_lines=label_reader.readlines()
        if ins_lines:
            list(map(cal_bycls,ins_lines))
    print('T num: {}\nZ num: {}\nP num: {}\n\ntotal: {}\n'.format(*cls_num,sum(cls_num)))