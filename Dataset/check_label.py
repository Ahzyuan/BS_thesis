import os,cv2,shutil
from glob import glob
from tqdm import tqdm

img_dir=r'Dataset\pick_img'
label_dir=r'Dataset\labels_darklabel'
save_dir=r'E:\AIL\project\Undergraduate_Thesis\Dataset\check_label\label'

label_list=glob(label_dir+'\\*')
cls_idx=['T','Z','P']

for label in tqdm(label_list):
    label_name=os.path.split(label)[-1]
    co_img_path=os.path.join(img_dir, label_name.replace('.txt','.png'))
    co_img=cv2.imread(co_img_path)
    h,w=co_img.shape[:2]
    
    with open(label,'r',encoding='utf-8') as label_reader:
        objs=label_reader.readlines()
    
    new_objs=[]
   
    for obj in objs:
        obj=obj.strip()
        cls_id, nor_cenx, nor_ceny, nor_w, nor_h=obj.split(' ')
                
        cen_x=float(nor_cenx)*w
        cen_y=float(nor_ceny)*h
        obj_w=float(nor_w)*w
        obj_h=float(nor_h)*h
        
        x_min=max(round(cen_x-obj_w/2),0)   # correct the exceeding box
        x_max=min(round(cen_x+obj_w/2),w)
        y_min=max(round(cen_y-obj_h/2),0)
        y_max=min(round(cen_y+obj_h/2),h)

        nor_cenx='{:.7f}'.format(0.5*(x_max+x_min)/w)
        nor_ceny='{:.7f}'.format(0.5*(y_max+y_min)/h)
        nor_w='{:.7f}'.format((x_max-x_min)/w)
        nor_h='{:.7f}'.format((y_max-y_min)/h)
        
        if cls_id not in ['0','1','2']:     # correct the wrong class idx
            print('\r{}\tcls_id {}'.format(label_name,cls_id))
            
            left_top=(x_min, y_min)
            right_buttom=(x_max, y_max)
            
            cv2.rectangle(co_img, left_top, right_buttom, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow('error_box',co_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            correct_label=input('The correct label of this box is: ')
            try:
                cls_id=cls_idx.index(correct_label)
            except:     # discard this obj
                continue
        
        new_objs.append(' '.join([str(cls_id), nor_cenx, nor_ceny, nor_w, nor_h]))

    save_path=os.path.join(save_dir, label_name)
    with open(save_path,'w',encoding='utf-8') as label_writer:
        label_writer.writelines('\n'.join(new_objs))
    #shutil.copyfile(co_img_path,os.path.join(r'E:\AIL\project\Undergraduate_Thesis\Dataset\check_label\img',label_name.replace('.txt','.png')))

