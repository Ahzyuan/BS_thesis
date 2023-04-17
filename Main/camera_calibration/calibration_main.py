import cv2, glob, os, sys, shutil
import numpy as np
from tqdm import tqdm
 
def camera_calibration(img_dir, type='chess', npy_save_path=sys.path[0], draw_corners=False):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  #(2+1, 30, 0.001)
    print('please input the size of the calibration board:')
    inner_grid_w = int(input('column:  '))-1
    inner_grid_h = int(input('row:  '))-1

    # initialize the world coordinate system, generate each point's coordinate in the plane Z=0 in the world coordinate system
    world_init_p = np.zeros((inner_grid_w*inner_grid_h, 3), np.float32)
    world_init_p[:,:2] = np.mgrid[0:inner_grid_w, 0:inner_grid_h].T.reshape(-1,2)
    world_points = [] 
    img_points = []
    img_path_list=[]    # restore the abs path of valid imgs
    
    if draw_corners:
        draw_imgs_save_path=os.path.join(img_dir,'draw_corners')
        if os.path.exists(draw_imgs_save_path):
            shutil.rmtree(draw_imgs_save_path)
        os.mkdir(draw_imgs_save_path)

    img_paths = glob.glob(img_dir+'\\*')
    for img_path in img_paths:
        if os.path.splitext(img_path)[-1] in ['.png','.jpg','.jpeg','.bmp']:
            
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_name=os.path.split(img_path)[-1]
            
            if type=='chess':
                ret, corners = cv2.findChessboardCorners(gray_img, (inner_grid_w, inner_grid_h), None)
            elif type=='circle':
                params = cv2.SimpleBlobDetector_Params()
                params.maxArea = 1e5
                params.minArea = 10
                params.minDistBetweenBlobs = 5
                blobDetector = cv2.SimpleBlobDetector_create(params)
                ret, corners = cv2.findCirclesGrid(gray_img, (inner_grid_w, inner_grid_h), cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector, None)
            else:
                assert 1==0,'now calibration plate\'s type is "{}", please check !'.format(type) 
            
            if ret == True:
                corners_refined=cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
                world_points.append(world_init_p)
                img_points.append(corners_refined.reshape([-1, 2]))
                img_path_list.append(img_path)

                if draw_corners:
                    #drawing for 3 times
                    cv2.drawChessboardCorners(img, (inner_grid_w, inner_grid_h), corners, ret)
                    cv2.drawChessboardCorners(img, (inner_grid_w, inner_grid_h), corners, ret)
                    cv2.drawChessboardCorners(img, (inner_grid_w, inner_grid_h), corners, ret)

                    cv2.imwrite(os.path.join(draw_imgs_save_path, img_name),img)
    
    valid_num=len(world_points)
    if valid_num==0:  
        assert 1==0,'can not detect calibration board !'
    else:
        print('valid imgs\' num: {}'.format(valid_num))

    ret, intri_mat, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, gray_img.shape[::-1], None, None, criteria=criteria)
    
    extri_mat=np.zeros((len(rvecs),4,4))
    for idx,i in enumerate(rvecs):
        mR, _ = cv2.Rodrigues(i)
        mT = tvecs[idx]
        exmat = np.concatenate([mR, mT], axis=1)
        exmat = np.vstack([exmat, np.array([0, 0, 0, 1])])  # 外参矩阵
        #exmat = np.expand_dims(exmat, 0)
        extri_mat[idx] = exmat
    
    np.save(os.path.join(npy_save_path,'intri_mat.npy'), intri_mat)
    np.save(os.path.join(img_dir,'intri_mat.npy'), intri_mat)
    #forestraight_pic_idx=int(input('the index of the forestraight picture, start from 1: '))
    #np.save(os.path.join(save_path,'extri_mat.npy'), extri_mat[forestraight_pic_idx-1])

    return img_path_list, world_points, img_points, intri_mat, dist, rvecs, tvecs
 
def undistort(img_path, mtx, dist, is_crop=False):
    
    img = cv2.imread(img_path)
    h,  w = img.shape[:2]
    
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h)) # 自由比例参数
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    if is_crop:
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    cv2.imshow('undistort img',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('calibresult.png',dst)
 
def get_reproject_error(world_points, img_points,  mtx, dist, rvecs, tvecs, img_path_list=None):
    error_collect = np.zeros(len(world_points))

    for img_idx, _ in enumerate(world_points):
        imgpoints2, _ = cv2.projectPoints(world_points[img_idx], rvecs[img_idx], tvecs[img_idx], mtx, dist)
        error = cv2.norm(img_points[img_idx],imgpoints2.reshape(-1,2), cv2.NORM_L2)/len(imgpoints2)
        error_collect[img_idx] = error

    total_error=error_collect.sum().item()
    mean_error=error_collect.mean().item()
    error_report="total error: {:.3f}\nmean_error: {:.3f}".format(total_error,mean_error)
    print(error_report)

    if img_path_list:
        error_log_path=os.path.split(img_path_list[0])[0]
        res_str=''
        
        for max_error_idx in np.argsort(error_collect)[::-1]:
            res_str+='{}\t error {}\n'.format(os.path.split(img_path_list[max_error_idx])[-1],error_collect[max_error_idx])
        res_str+='\n'+error_report
        
        with open(os.path.join(error_log_path,'error_log.txt'),'w',encoding='utf-8') as error_log_writer:
            error_log_writer.writelines(res_str)

    return total_error, mean_error

def correct_img_size(all_imgs_path, scale_rate=1):
    hor_img_dict={}
    ver_img_dict={}
    img_path_list=glob.glob(all_imgs_path+'\\*')

    save_path=os.path.join(all_imgs_path, 'resize_img')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    for img_path in tqdm(img_path_list, desc='Resizing'):
        if os.path.splitext(img_path)[-1] in ['.png','.jpg','.jpeg','.bmp']:
            img=cv2.imread(img_path)
            img_h,img_w=img.shape[:2]
            img_name=os.path.split(img_path)[-1]

            img_resize=cv2.resize(img, (int(img_w*scale_rate),int(img_h*scale_rate)))
            cv2.imwrite(os.path.join(save_path,img_name), img_resize)

            if img_h < img_w:   # this img is horizontal
                hor_img_dict[img_name]=img_resize
            else:
               ver_img_dict[img_name]=img_resize

    major_orientation=1 if len(hor_img_dict)>=len(ver_img_dict) else 0 # 1:most of the imgs are horizontal, 0 for the otherwise meanings
        
    if major_orientation==1:
        for ver_img_name, ver_img in ver_img_dict.items(): 
            img_rotate=cv2.rotate(ver_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(save_path,ver_img_name), img_rotate)
    else:
        for hor_img_name, hor_img in hor_img_dict.items(): 
            img_rotate=cv2.rotate(hor_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(save_path,hor_img_name), img_rotate)
    return save_path

if __name__=='__main__':

    img_dir=r'E:\AIL\project\Undergraduate_Thesis\Main\camera_calibration\img_dir\1080\A3_night'
    assert os.path.exists(img_dir),'The path "{}" doesn\'t exist, please check !'.format(img_dir)

    img_dir=correct_img_size(img_dir,scale_rate=1)
    img_path_list,world_points, img_points, mtx, dist, rvecs, tvecs=camera_calibration(img_dir,draw_corners=True)
    #undistort(os.path.join(img_dir,'8.jpg'), mtx, dist, is_crop=True)
    total_error, mean_error = get_reproject_error(world_points, img_points, mtx, dist, rvecs, tvecs, img_path_list=img_path_list)
