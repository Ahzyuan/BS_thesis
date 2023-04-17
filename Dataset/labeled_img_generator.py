import os,cv2
from glob import glob
from tqdm import tqdm

img_dir=r'Dataset\pick_img'
label_dir=r'Dataset\labels'
save_dir=r'Dataset\labeled_imgs'

img_list=glob(img_dir+'\\*')

type_num=['T','Z','P']
type_color=[[151, 46, 240],[240, 174, 46],[46,124,240]]

count=0
for solo_img in tqdm(img_list):
    count+=1
    img_name=os.path.split(solo_img)[-1]
    co_label=os.path.splitext(img_name)[0]+'.txt'
    img_path=solo_img
    co_label_path=os.path.join(label_dir, co_label)

    save_path=os.path.join(save_dir, str(count)+'.png')

    img=cv2.imread(img_path)
    h,w=img.shape[:2]
    
    if os.path.exists(co_label_path):
        with open(co_label_path,'r',encoding='utf-8') as label_reader:
            objs=label_reader.readlines()

        if objs:
            for obj in objs:
                obj=obj.strip()
                type_idx, nor_cenx, nor_ceny, nor_w, nor_h=obj.split(' ')
                type_idx=int(type_idx)
                obj_type=type_num[type_idx]
                obj_color=type_color[type_idx]

                cen_x = float(nor_cenx) * float(w)
                cen_y = float(nor_ceny) * float(h)
                obj_w = float(nor_w) * float(w)
                obj_h = float(nor_h) * float(h)

                left_top=(int(cen_x-obj_w/2), int(cen_y-obj_h/2))
                right_buttom=(int(cen_x+obj_w/2), int(cen_y+obj_h/2))
                text_loc=(left_top[0],left_top[1]-2)

                cv2.rectangle(img, left_top, right_buttom, obj_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img, obj_type, left_top, 0, 0.5, obj_color, thickness=1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(save_path, img)
