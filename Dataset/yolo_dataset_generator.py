import os, random, shutil, sys, yaml
from tqdm import tqdm

program_root_path = sys.path[0]
img_root_path = os.path.join(program_root_path, 'pick_img')
label_root_path = os.path.join(program_root_path, 'labels')
yolo_root_path = os.path.join(os.path.dirname(program_root_path), 'yolov8') # ../yolov8
assert os.path.exists(yolo_root_path), 'yolo project dir not found'

img_list = os.listdir(img_root_path)
val_rate = 0.1
test_rate = 0.1

# create dir (if datsaset already exists, the files within it will be rewrited)
dataset_path = os.path.join(yolo_root_path,'TPZ_VOCdevkit')
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

train_img_path = os.path.join(dataset_path,'train','images')
val_img_path = os.path.join(dataset_path,'val','images')
test_img_path = os.path.join(dataset_path,'test','images')

train_label_path = os.path.join(dataset_path,'train','labels')
val_label_path = os.path.join(dataset_path,'val','labels')
test_label_path = os.path.join(dataset_path,'test','labels')
path_alley=[train_img_path, val_img_path, test_img_path, 
            train_label_path, val_label_path, test_label_path]
for path in path_alley:
    if not os.path.exists(path):
        os.makedirs(path)

#pick test & val img
val_img_num = round(len(img_list)*val_rate)
test_img_num = round(len(img_list)*test_rate)
val_img_list = random.sample(img_list, val_img_num)
test_img_list = random.sample(set(img_list)-set(val_img_list), test_img_num)

#copy img and corresponding processed label
for img in tqdm(img_list):
    label_name = os.path.splitext(img)[0]+'.txt'
    
    #generate moving path
    origin_img_path = os.path.join(img_root_path, img)
    origin_label_path = os.path.join(label_root_path, label_name)
    
    with open(origin_label_path,'r',encoding='utf-8') as label_reader:
            labels = label_reader.readlines()

    if img in test_img_list:
        img_des_path = test_img_path
        label_des_path =  test_label_path
        log_des_path = os.path.join(dataset_path,'test.txt')
    elif img in val_img_list:
        img_des_path = val_img_path
        label_des_path =  val_label_path
        log_des_path = os.path.join(dataset_path,'val.txt')
    else:
        img_des_path = train_img_path 
        label_des_path = train_label_path 
        log_des_path = os.path.join(dataset_path,'train.txt') 

    new_img_path = os.path.join(img_des_path, img)
    new_label_path = os.path.join(label_des_path, label_name)
    
    with open(log_des_path,'a',encoding='utf-8') as log_writer:
        log_writer.write(new_img_path+'\n')
    
    #copying
    shutil.copyfile(origin_img_path, new_img_path)
    shutil.copyfile(origin_label_path, new_label_path)

yaml_data = {
    'train': os.path.join(dataset_path,'train.txt') ,
    'val': os.path.join(dataset_path,'val.txt') ,
    'test': os.path.join(dataset_path,'test.txt') ,
    'nc': 3,
    'names': ['T','Z','P']
}

with open(os.path.join(dataset_path,'TPZ.yaml'), 'w') as file:
    yaml.dump(yaml_data, file, allow_unicode=True) 
