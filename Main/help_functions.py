import cv2,os,shutil,torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from time import time
from torchvision.ops import nms

from config import config

# Dataset related
def gap_pick(origin_path, target_num=12000):
    assert os.path.exists(origin_path), f'pictures\' original path {origin_path} doesn\'t exist'
    new_path=os.path.split(origin_path)[0]+'\\pick_img'
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)

    img_list=os.listdir(origin_path)[::-1]
    origin_num=len(img_list)
    
    gap=round((origin_num/(target_num-1))-1)
    if gap<1:
        gap=1

    idx_generate=(gap+1)*(np.arange(1,origin_num+1)-1)  #(gap+1)*(np.arange(1,target_num+1)-1)
    idx_generate=idx_generate[idx_generate<=origin_num]

    img_list=np.array(img_list)
    pick_list=img_list[idx_generate]

    for img in tqdm(pick_list):
        img_origin_path=os.path.join(origin_path,img)
        img_new_path=os.path.join(new_path,img)
        shutil.copyfile(img_origin_path,img_new_path)

def match_pick_label(img_path,label_path):
    img_list='\n'.join(os.listdir(img_path)).replace('.png','.txt')
    img_list=set(img_list.split('\n'))
    label_list=set(os.listdir(label_path))

    match_label=label_list-img_list

    for label in tqdm(match_label):
        os.remove(os.path.join(label_path,label))

def recorrect_label(copy_label_path=r'E:\AIL\project\Undergraduate_Thesis\VOC2007\error_label'):
    label_list=glob(copy_label_path+'\\*')
    
    for label in label_list:
        with open(label,'r',encoding='utf-8') as f1:
            item_list=f1.readlines()
        
        new_item=[]
        for item in item_list:
            item=item.strip()
            item_sep=item.split(' ')
            if float(item_sep[2])*1080>=535 and item_sep[0]=='0':
                item_sep[0]='1'
            new_item.append(' '.join(item_sep))
        new_log='\n'.join(new_item)
        with open(label,'w',encoding='utf-8') as f2:
            f2.writelines(new_log)

def cal_instance_num(label_path):
    label_list=glob(label_path+'\\*')
    cls_num=[0,0,0] #num_T, num_Z, num_P

    def cal_bycls(line):
        cls_idx=int(line.strip()[0])
        cls_num[cls_idx]+=1

    for label in label_list:
        with open(label,'r',encoding='utf-8') as label_reader:
            ins_lines=label_reader.readlines()
        if ins_lines:
            list(map(cal_bycls,ins_lines))
    print('T num: {}\nZ num: {}\nP num: {}\n\ntotal: {}\n'.format(*cls_num,sum(cls_num)))

# detect_api related
def get_IPM_dis(im0, obj_yr):
    h,w=im0.shape[:2]
    tl_func=lambda y: (round(w*(h/2-y)/h+w/2), y)
    tr_func=lambda y: (round(w*(y-h/2)/h+w/2), y)
    tl,tr=tl_func(obj_yr), tr_func(obj_yr)
    bl,br=(0,h), (w,h)
    origin_p=np.float32([tl,tr,bl,br])

    asure_loc=round(754*h/1080)
    cv2.line(im0,(0,asure_loc),(w,asure_loc),color=(242,12,12),thickness=2)

    IPM_width=500
    IPM_height=500
    IPM_p=np.float32([[IPM_width/2-IPM_width/10, 0],
                      [IPM_width/2+IPM_width/10, 0],
                      [IPM_width/2-IPM_width/10, IPM_height],
                      [IPM_width/2+IPM_width/10, IPM_height]])
    matrix = cv2.getPerspectiveTransform(origin_p, IPM_p)
    output = cv2.warpPerspective(im0, matrix, (IPM_width, IPM_height))

    output_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(output_hsv, np.array([120,43,46]), np.array([124,255,255]))
    line_loc=np.argmax(np.sum(mask,axis=1))
    dis=IPM_height*8.636/(IPM_height-line_loc)  # 8.636 is the real dis for the asure_line
    return dis

def get_cls_distance(im0, tar_obj_dict, intri_mat):
    fy=intri_mat[1,1]
    frame_h=im0.shape[0]
    v0=intri_mat[1,2]
    H=1.3
    # cls_info: x_l, y_l, x_r, y_r, conf, cls, cen_x, y_r
    for obj_cls, cls_info in tar_obj_dict.items():
        if obj_cls =='T':
            criterion=(min(cls_info[2]-cls_info[0], cls_info[3]-cls_info[1]))*1080/frame_h # use the shortest edge to calculate dis
            tar_obj_dict[obj_cls]=np.append(cls_info,max(fy*0.35/criterion-1.8,0))  # unit meter, 0.35m for Φ300 traffic light, 1.8m is the dis from the camera to the front of the car
        elif obj_cls=='P':
            criterion=(cls_info[3]-cls_info[1])*1080/frame_h # use the height of person to calculate dis
            tar_obj_dict[obj_cls]=np.append(cls_info,max(fy*1.64/criterion-1.8,0))  # unit meter, default person height is 1.64m
        else:
            #tar_obj_dict[obj_cls]=np.append(cls_info,max(7098.6*(cls_info[3]*1080/frame_h-v0)**(-1.249)-1.8,0))
            #tar_obj_dict[obj_cls]=np.append(cls_info,max(get_IPM_dis(im0, cls_info[3])-1.8,0))
            tar_obj_dict[obj_cls]=np.append(cls_info,max(fy*H/(abs(cls_info[3]-frame_h//2)*1080/frame_h)-1.8,0))
    
    # tar_obj_dict: x_l, y_l, x_r, y_r, conf, cls, cen_x, y_r, dis
    return tar_obj_dict

def get_focus_dis(det, obj_names, valid_h, valid_w, valid_zw): # det [[]]: x_l, y_l, x_r, y_r, conf, cls
    if type(det)==torch.Tensor:
        objs_info=det.cpu().numpy()
    
    objs_info=np.c_[objs_info, 
                    (objs_info[:,0]+objs_info[:,2])/2, # bbox center x
                    objs_info[:,3]]     # bbox buttom y
    
    grey_objs_idx=np.where(objs_info[:,6]<=valid_w[0])[0].tolist()+\
                  np.where(objs_info[:,6]>=valid_w[1])[0].tolist()+\
                  np.where(objs_info[:,3]<=valid_h)[0].tolist()
    grey_objs=objs_info[grey_objs_idx,:]    #the invalid objs: x_l, y_l, x_r, y_r, conf, cls, cen_x, y_r
    
    objs_info=np.delete(objs_info, grey_objs_idx, axis=0) # pick valid item

    #objs_info: x_l, y_l, x_r, y_r, conf, cls, cen_x, y_r
    tar_obj_dict={}
    else_objs=np.array([])
    for cls_id,cls in enumerate(obj_names):
        upper_cls=cls.upper()
        cls_row_idx=np.where(objs_info[:,5]==cls_id)[0]

        if cls_row_idx.any():   # if exists this cls
            cls_mat=objs_info[cls_row_idx, :]
            if  upper_cls =='P':
                target_idx=np.argmax(cls_mat[:,-1])
            elif upper_cls=='Z':
                invalid_z_idx=np.where(cls_mat[:,6]<=valid_zw[0])[0].tolist()+\
                               np.where(cls_mat[:,6]>=valid_zw[1])[0].tolist()
                if invalid_z_idx:
                    grey_objs=np.r_[grey_objs, cls_mat[invalid_z_idx,:]] if grey_objs.any() else cls_mat[invalid_z_idx,:]
                
                cls_mat=np.delete(cls_mat, invalid_z_idx, axis=0)
                if cls_mat.any():
                    target_idx=np.argmax(cls_mat[:,-1])
                else:
                    continue
            else:
                target_idx=np.argmin(cls_mat[:,1])

            tar_obj_dict[upper_cls]=cls_mat[target_idx]
            if len(else_objs)==0:
                else_objs=np.delete(cls_mat,target_idx,axis=0)
            else:
                if len(cls_mat)>1:
                    else_objs=np.r_[else_objs, np.delete(cls_mat,target_idx,axis=0)] 
    
    if grey_objs.any():
        grey_objs=grey_objs[:,:6]
        grey_objs[:,5]=len(obj_names)   # change the cls to 4 ,in order to pick the grey color later
    
    if else_objs.any():
        else_objs=else_objs[:,:6]

    return tar_obj_dict, grey_objs, else_objs

def get_light_color(roi):
    roi_gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask_ths=cv2.threshold(roi_gray, 147, 255, cv2.THRESH_BINARY)
    ths_img=cv2.bitwise_and(roi,roi,mask=mask_ths)
    cv2.imwrite(r'E:\AIL\project\Undergraduate_Thesis\source\color_thred\ths\img.png',ths_img)
    ths_img_hsv = cv2.cvtColor(ths_img, cv2.COLOR_BGR2HSV)
    
    possible_colors={}

    for color,(min_list,max_list) in config['hsv_color_range'].items():
        mask=cv2.inRange(ths_img_hsv, min_list, max_list)
        #mask=cv2.medianBlur(mask, 3)   # erode the single pixel
        
        #mask=cv2.bitwise_and(roi,roi,mask=mask)
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)
        #cv2.destroyWindow('mask')

        mask=mask//255
        roi_ratio=np.sum(mask)/mask.size
        
        if 0.01 < roi_ratio < 0.33:
            possible_colors[roi_ratio]=color
           
    pred_color=possible_colors[max(possible_colors)][0] if possible_colors else 'B'  # get the first character of the color
    
    return pred_color

# yolov5 function
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

if __name__=='__main__':
    #gap_pick(r'E:\AIL\project\Undergraduate_Thesis\VOC2007\images')
    #match_pick_label(r'E:\AIL\project\Undergraduate_Thesis\VOC2007\pick_img',r'E:\AIL\project\Undergraduate_Thesis\VOC2007\labels_darklabel')
    #cal_instance_num(r'E:\AIL\project\Undergraduate_Thesis\Dataset\labels_test')
    #img_path=r'E:\AIL\project\Undergraduate_Thesis\source\color_thred\light_img\8.png'
    #img=cv2.imread(img_path)
    #_=get_light_color(img)
    pass