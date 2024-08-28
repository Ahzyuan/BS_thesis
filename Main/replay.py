import os,cv2,argparse,yaml,math,sys,shutil
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from PIL import Image, ImageDraw, ImageFont, ImageFilter

last_shape = (288,480,3)
red_masks = {}
font_file = os.path.join(os.path.dirname(sys.path[0]), 'Assets', 'consolai.ttf')
assert os.path.exists(font_file), f'font file {font_file} not found'

def update_args(args):
    with open(args.config, 'r') as cfg_reader:
        config = yaml.safe_load(cfg_reader) # dict
    
    args.__dict__.update(config.get("det_analyse_settings", dict()))
    args.names = config['names']
    args.max_a = 250 / (9*config.get("min_zero2hundred_time", 4)) # unit m/s^2
    
    return args

def draw_box(img, box_data, line_width=None):
    '''
    img: ndarray
    box_data: (x1, y1, x2, y2, cls_name, box_color_idx, traffic_color, depth)
    '''
    global args

    x1, y1, x2, y2, cls, box_color_idx, traffic_color = map(int, box_data[:-1])

    cls_name = args.names[cls]
    depth = f'{box_data[-1]:.2f}' if box_data[-1] else None
    box_color = args.box_color[box_color_idx]

    lw = line_width or round(sum(img.shape) / 2 * 0.003)
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale
    
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    
    if box_color_idx == -1: # emergency object
        label = f' {cls_name} {depth} m' if cls_name!='T' else f' {cls_name} {chr(traffic_color)} {depth} m'
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        h += 3  # add pixels to pad text
        
        outside = y1 >= h  # label fits outside box
        box_buttom_mid = (int((x1+x2)//2), y2)
        x1 = min(x1, img.shape[1] - w) # shape is (h, w), check if label extend beyond right side of image
        x2 = x1 + w
        y2 = y1 - h if outside else y1 + h

        cv2.rectangle(img, (x1,y1), (x2, y2), box_color, -1, cv2.LINE_AA)  # filled
        
        cv2.putText(
            img,
            label,
            (x1, y1 - 2 if outside else y1 + h - 1),
            0,
            sf,
            [255,255,255], # text color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

        cv2.line(img,
                 (int(img.shape[1]//2), img.shape[0]), 
                 box_buttom_mid, 
                 (0, 0, 255))
    
    return img

def put_detect_data(cv_img, detect_data):
    '''
    cv_img: ndarray
    detect_data: dict, keys: ['T':(depth, color), 'Z/P':(depth), 'fps', 'v0', 'brake_a']
    '''
    global font_file
    Tcolor_dict={
        'R':'#FF3C3C',
        'G':'#00FF80',
        'B':'#FFFFFF',
        'Y':'#FFF206',
        'U':'#FFFFFF' # when T is not detected
    }

    frame_h, frame_w = cv_img.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    pil_drawing = ImageDraw.Draw(pil_img)

    fps_str = 'FPS '+ str(detect_data['fps'])
    sep_str = '-'*12
    P_str = 'P '+ detect_data.get('P', ['None'])[0] +' m'
    Z_str = 'Z '+ detect_data.get('Z', ['None'])[0] +' m'
    T_str = 'T '+ detect_data.get('T', ['None'])[0] +' m'
    v_str = 'V  '+ detect_data['v0'] +' km/h'
    a_val = 'aˉ '+ detect_data['brake_a'] +' m/s²'
    T_color = Tcolor_dict[detect_data.get('T', ['None','U'])[1]]
        
    log_str_up = '\n'.join([fps_str, 
                            sep_str, 
                            v_str, 
                            a_val, 
                            sep_str, 
                            P_str, 
                            Z_str])
    log_str_down = T_str
    
    text_size = round(frame_h*20/543)
    line_space = int(text_size/2)
    up_xl,up_yl,up_xr,up_yr = pil_drawing.textbbox((0,0), log_str_up, spacing=line_space, font=ImageFont.truetype(font_file, text_size))
    down_xl,down_yl,down_xr,down_yr = pil_drawing.textbbox((0,0), log_str_down, spacing=line_space, font=ImageFont.truetype(font_file, text_size))
    
    # text total size
    xl = up_xl
    xr = max(up_xr, down_xr)
    yl = up_yl
    yr = up_yr + line_space + down_yr-down_yl
    
    # text patch cords
    margin = round(frame_h*0.015)
    patch_yl = margin
    patch_xl = frame_w - margin*3 - xr + xl
    patch_xr = frame_w - margin
    patch_yr = patch_yl + yr - yl + margin*2

    pil_patch = Image.fromarray(cv2.cvtColor(cv_img[patch_yl:patch_yr+1, patch_xl:patch_xr+1,:], cv2.COLOR_BGR2RGB))
    pil_patch = pil_patch.filter(ImageFilter.GaussianBlur(radius=4))
    pil_drawing = ImageDraw.Draw(pil_patch)

    # put text on patch
    loc_up = (margin, margin)
    loc_down = (margin, margin + up_yr-up_yl + line_space)
    pil_drawing.text(loc_up, log_str_up, align='left',fill=(255, 255, 255), font=ImageFont.truetype("consolai.ttf", text_size), spacing=line_space)
    pil_drawing.text(loc_down, log_str_down, align='left', fill=T_color, font=ImageFont.truetype("consolai.ttf", text_size))
    
    patch = cv2.cvtColor(np.asarray(pil_patch), cv2.COLOR_RGB2BGR)

    return pil_img, patch, (patch_xl, patch_yl, patch_xr, patch_yr)

def get_red_mask(img_shape, opacity=128):
    '''
    img_shape: (w, h)
    opacity = 128  # 0-255
    '''
    global red_masks
    
    mask_key = '{}x{}'.format(*img_shape)
    if mask_key in red_masks:
        for mask in red_masks[mask_key]:
            mask.putalpha(opacity)

    masks = deque([])
    for angle in range(-120,-59, 5):
        w,h = img_shape
        ratio = w/h
        w_border = math.sin(math.radians(angle))/10
        h_border = w_border*ratio

        mask = Image.new("L", img_shape, opacity)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([(int(w*w_border), int(h*h_border)), 
                    (int(w*(1-w_border)), int(h*(1-h_border)))], fill=255)
        
        masks.append(mask)
    
    red_masks[mask_key] = masks

def draw_warning(img, feather_radius=70):
    '''
    img: PIL.Image, RGB
    '''
    global red_masks

    image = img.convert("RGBA")
    width, height = image.size

    red_layer = Image.new("RGBA", (width, height), (255, 0, 0, 255))

    mask_key = f'{width}x{height}'
    mask = deepcopy(red_masks[mask_key][0])
    red_masks[mask_key].rotate(-1)

    mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))
    mask = mask.point(lambda p: 255 - p) # reverse pattern

    red_layer.putalpha(mask) # merge mask and red_layer

    combined_image = Image.alpha_composite(image, red_layer) # put red_layer on img
    return cv2.cvtColor(np.asarray(combined_image), cv2.COLOR_RGB2BGR)

def combine_img_meta(img_path, meta_path):
    global last_shape, args

    img = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros(last_shape, dtype=np.uint8)
    if args.scale != 1:
        img = cv2.resize(img, (int(img.shape[1]*args.scale), int(img.shape[0]*args.scale)))
    frame_shape = img.shape
    last_shape = frame_shape

    meta = np.load(meta_path) # [x1, y1, x2, y2, (track_id), conf, cls, box_color_idx, traffic_color, depth, fps, v0, brake_a, if_alert]

    if meta.any():
        # draw boxes
        emergency_info = {}
        meta[:,:4] = meta[:,:4] * args.scale
        for box in meta:
            x1, y1, x2, y2, *_, cls, box_color_idx, traffic_color_ascii, depth, fps, v0, brake_a, if_alert = box
            
            cls_name = args.names[int(cls)]
            depth = depth if depth < 2000 else None
            traffic_color = chr(int(traffic_color_ascii))

            img = draw_box(img, [x1, y1, x2, y2, cls, box_color_idx, traffic_color_ascii, depth])

            if box_color_idx == -1: # emergency object
                emergency_info[cls_name] = [f'{depth:.2f}' if depth else 'None'] + \
                                           ([] if cls_name != 'T' else [traffic_color]) # depth, color
               
        # put detailed detected data on the right top region of img
        emergency_info['fps'] = int(fps)
        emergency_info['v0'] = f'{v0:.2f}' if not np.isnan(v0) else 'None'
        emergency_info['brake_a'] = f'{brake_a:.2f}'
        pil_img, cv_patch, patch_xyxy = put_detect_data(img, emergency_info)

        # alert
        if if_alert:
            alert_rate = brake_a / args.max_a
            get_red_mask((frame_shape[1],frame_shape[0]), int(128*alert_rate))
            img = draw_warning(pil_img)

        patch_xl, patch_yl, patch_xr, patch_yr = patch_xyxy
        img[patch_yl:patch_yr+1, patch_xl:patch_xr+1, :] = cv_patch

    return img, fps

def main(root, save_path):
    '''
    root: the result dir, compose of frame image dir `img` and frame meta info dir `meta`.
          In `img` dir, each frame image is named as `image{frame_id}.jpg`, 
          and `meta` dir contains corresponding `image{frame_id}.npy` file, which is a matrix of shape (boxes_num, 14(13)).
          The meta matrix columns are: [x1, y1, x2, y2, (track_id), conf, cls, box_color_idx, traffic_color, depth, fps, v0, brake_a, if_alert]
    
    save_path: the path to save the replace result. If is a path pointing to a dir, it will save the each frame.
               If is a path pointing to a mp4 file, it will save the whole video.
    '''
    seq_name = os.path.basename(root)
    img_dir = os.path.join(root, 'img')
    meta_dir = os.path.join(root, 'meta')

    img_files = sorted(os.listdir(img_dir))

    save_vid = 0
    if save_path:        
        base = os.path.basename(save_path)
        if '.' in base: # save as video
            save_vid = 1
            save_path = os.path.splitext(save_path)[0]
        os.makedirs(save_path, exist_ok=True)

    frame_ls, fps_ls = [], []
    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        meta_name = f'{img_name}.npy'
        
        img, fps = combine_img_meta(os.path.join(img_dir, img_file), # ndarray
                                               os.path.join(meta_dir, meta_name))

        img_save_path = os.path.join(save_path, img_file)
        cv2.imwrite(img_save_path, img)
        if save_vid:
            frame_ls.append(img_save_path)
            fps_ls.append(fps)

        frame_interval = int(1000/fps)
        cv2.imshow(seq_name, img)
        if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = round(sum(fps_ls) / len(fps_ls))
        vid = cv2.VideoWriter(save_path+'.mp4', fourcc, fps, (img.shape[1], img.shape[0]))
        for frame in tqdm(frame_ls, desc='Saving video'):
            vid.write(cv2.imread(frame))
        vid.release()
        shutil.rmtree(save_path)
        print('Done.\nExit.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--res_dir', type=str, required=True, help='Recored sequence dir')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='Recored sequence dir')
    parser.add_argument('-c', '--config', type=str, default='TZP.yaml', help='config file path')
    parser.add_argument('--scale', type=int, default=1, help='scale the image')
    args = parser.parse_args()
    args.res_dir = os.path.abspath(args.res_dir)
    args.save_path = os.path.abspath(args.save_path)
    args.config = os.path.abspath(args.config)
    assert os.path.exists(args.res_dir), f"{args.res_dir} not exists"

    args = update_args(args)

    main(args.res_dir, args.save_path)
