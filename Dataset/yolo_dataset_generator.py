import os, random, shutil
from tqdm import tqdm

img_root_path=r'Dataset\pick_img'
label_root_path=r'Dataset\labels'
save_path=r'Main\yolov5'
program_root_path=os.getcwd()

img_list=os.listdir(img_root_path)
split_rate=0.2

# create dir (if datsaset already exists, the files within it will be rewrited)
dataset_path=os.path.join(save_path,'{}_VOCdevkit'.format('TPZ'))
server_dataset_path='/autodl-tmp/yolo5/'+'{}_VOCdevkit'.format('TPZ')
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

train_img_path=os.path.join(dataset_path,'images','train')
test_img_path=os.path.join(dataset_path,'images','val')
server_train_img_path=server_dataset_path+'/images/train'
server_test_img_path=server_dataset_path+'/images/val'

train_label_path=os.path.join(dataset_path,'labels','train')
test_label_path=os.path.join(dataset_path,'labels','val')
path_alley=[train_img_path, test_img_path, train_label_path, test_label_path]
for path in path_alley:
    if not os.path.exists(path):
        os.makedirs(path)

#pick test img
test_img_num=round(len(img_list)*split_rate)
test_img_list = random.sample(img_list, test_img_num)

#copy img and corresponding processed label
for img in tqdm(img_list):
    label_name=os.path.splitext(img)[0]+'.txt'
    
    #generate moving path
    origin_img_path=os.path.join(img_root_path, img)
    origin_label_path=os.path.join(label_root_path, label_name)
    
    with open(origin_label_path,'r',encoding='utf-8') as label_reader:
            labels=label_reader.readlines()
    
    img_des_path = test_img_path 
    server_img_des_path = server_test_img_path 
    label_des_path = test_label_path 
    log_des_path=os.path.join(dataset_path,'test.txt') 
    server_log_des_path=os.path.join(dataset_path,'test_server.txt') 
    
    if img not in test_img_list:
        img_des_path = train_img_path
        server_img_des_path =  server_train_img_path
        label_des_path =  train_label_path
        log_des_path=os.path.join(dataset_path,'train.txt')
        server_log_des_path=os.path.join(dataset_path,'train_server.txt')

    new_img_path=os.path.join(program_root_path, img_des_path, img)
    server_new_img_path=server_img_des_path+'/'+img
    new_label_path=os.path.join(program_root_path, label_des_path, label_name)
    
    with open(log_des_path,'a',encoding='utf-8') as test_img_writer:
        test_img_writer.write(new_img_path+'\n')
    with open(server_log_des_path,'a',encoding='utf-8') as server_test_img_writer:
        server_test_img_writer.write(server_new_img_path+'\n')
    
    #copying
    shutil.copyfile(origin_img_path, new_img_path)
    shutil.copyfile(origin_label_path, new_label_path)
