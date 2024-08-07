import os,sys
from tqdm import tqdm

pwd = sys.path[0]
img_dir = os.path.join(pwd, 'pick_img')
label_dir = os.path.join(pwd, 'labels')
save_dir = label_dir

img_allo=os.listdir(img_dir)
img_set=set([img.replace('.png','.txt') for img in img_allo])

label_set=set(os.listdir(label_dir))

lack_labels=img_set-label_set

for solo_label in tqdm(lack_labels):
    with open(os.path.join(save_dir,solo_label),'w',encoding='utf-8') as empty_writer:
        empty_writer.write('')
