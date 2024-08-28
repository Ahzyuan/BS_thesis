import os, random, shutil, sys, yaml, argparse
from tqdm import tqdm

def main(dataset_path, img_root, label_root, val_rate=0.2):
    # create dir (if datsaset already exists, the files within it will be rewrited)
    if os.path.exists(dataset_path): # rebuild dataset each time
        shutil.rmtree(dataset_path)

    train_img_path = os.path.join(dataset_path,'train','images')
    val_img_path = os.path.join(dataset_path,'val','images')
    train_label_path = os.path.join(dataset_path,'train','labels')
    val_label_path = os.path.join(dataset_path,'val','labels')

    for path in [train_img_path, val_img_path, train_label_path, val_label_path]:
        os.makedirs(path)

    #pick test & val img
    img_list = os.listdir(img_root)
    val_img_num = round(len(img_list)*val_rate)
    val_img_list = random.sample(img_list, val_img_num)

    #copy img and corresponding label
    for img in tqdm(img_list):
        label_name = os.path.splitext(img)[0]+'.txt'
        
        #generate moving path
        origin_img_path = os.path.join(img_root, img)
        origin_label_path = os.path.join(label_root, label_name)
        
        if img in val_img_list:
            img_des_path = val_img_path
            label_des_path =  val_label_path
        else:
            img_des_path = train_img_path 
            label_des_path = train_label_path 

        new_img_path = os.path.join(img_des_path, img)
        new_label_path = os.path.join(label_des_path, label_name)
              
        #copying
        shutil.copyfile(origin_img_path, new_img_path)
        shutil.copyfile(origin_label_path, new_label_path)
    
    # dumping dataset yaml file
    yaml_data = {
    'path': dataset_path ,
    'train': train_img_path ,
    'val': val_img_path ,
    'nc': 3,
    'names': {0:'T',1:'Z',2:'P'}
    }

    with open(os.path.join(dataset_path,'TPZ.yaml'), 'w') as file:
        yaml.dump(yaml_data, file, allow_unicode=True) 


if __name__ == '__main__':
    program_root_path = sys.path[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_root', type=str, default=os.path.join(program_root_path, 'pick_img'))
    parser.add_argument('-l', '--labels_root', type=str, default=os.path.join(program_root_path, 'labels'))
    parser.add_argument('-s', '--save_path', type=str, default=os.path.join(program_root_path, 'TPZ_VOCdevkit'))
    parser.add_argument('--val_rate', type=float, default=0.2)
    args = parser.parse_args()
    args.imgs_root = os.path.abspath(args.imgs_root)
    args.labels_root = os.path.abspath(args.labels_root)
    args.save_path = os.path.abspath(args.save_path)
    assert os.path.exists(args.imgs_root), 'imgs_root not exists'
    assert os.path.exists(args.labels_root), 'labels_root not exists'

    main(dataset_path = os.path.abspath(args.save_path),
         img_root = args.imgs_root,
         label_root = args.labels_root,
         val_rate = args.val_rate)