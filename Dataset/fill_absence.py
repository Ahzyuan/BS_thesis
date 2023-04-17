import os
from tqdm import tqdm

img_dir=r'Dataset\pick_img'
label_dir=r'Dataset\labels_darklabel'
save_dir=r'Dataset\labels'

img_allo='\n'.join(os.listdir(img_dir))
img_set=set(img_allo.replace('.png','.txt').split('\n'))

label_set=set(os.listdir(label_dir))

lack_labels=list(img_set-label_set)

for solo_label in tqdm(lack_labels):
    with open(os.path.join(save_dir,solo_label),'w',encoding='utf-8') as empty_writer:
        empty_writer.write('')
