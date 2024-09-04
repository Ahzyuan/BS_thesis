import os, rich,torch,cv2
import numpy as np
import pyrealsense2 as rs
from copy import deepcopy
from collections import Counter
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator
from ultralytics.data.loaders import LoadImagesAndVideos,LoadPilAndNumpy

class LoadRsCamera():
    def __init__(self, args):
        assert hasattr(args, "streams"), f"Please specify `streams` in {args.config}"
        
        self.config = rs.config()
        # enable stream defined in yaml file
        for stream_name, stream_args in args.streams.items():
            stream_args = {key: eval(value) for key, value in stream_args.items()}
            self.config.enable_stream(**stream_args)
        
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)
        
        device_metadata = self.profile.get_device()
        self.device_name = device_metadata.get_info(rs.camera_info.name)
        self.depth_sensor,self.color_sensor = device_metadata.query_sensors()
        self.depth_unit = self.depth_sensor.get_option(rs.option.depth_units) # Number of meters represented by a single depth unit
        
        self.filters = {}
        if hasattr(args, "advnc_args"):
            self.advance_tune(args.advnc_args)
        if hasattr(args, "post_process"):
            self.init_postprocess_filter(args.post_process)

        self.align = rs.align(rs.stream.color) # depth img align to color img
        
        self.bs = 1
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.count
    
    def __next__(self):
        '''
        cd_frame : list[np.ndarray], [color_img, depth_img(unit m)].
        '''
        while True:
            raw_frame = self.pipeline.wait_for_frames()
            state, *cd_frame = self.process(raw_frame)
            self.count += 1
            if state:
                # paths, imgs, info
                return [f'image{self.count}.jpg'], [cd_frame], [f'{self.device_name} frame{self.count}: ']
            else:
                rich.print(f'[bold red italic]WARNING ⚠️: {self.device_name} loss frame {self.count}![/bold red italic]')
        
    def process(self, raw_frame):
        state = True
        
        # align depth img to color img
        aligned_frames = self.align.process(raw_frame)
        color_part = aligned_frames.get_color_frame()
        depth_part = aligned_frames.get_depth_frame()

        if not color_part or not depth_part:
            state = False
            return state, None

        # post process
        if 'color' in self.filters:
            color_part = self.apply_filters_ls(color_part, self.filters['color'])
        if 'depth' in self.filters:
            depth_part = self.apply_filters_ls(depth_part, self.filters['depth'])
        
        color_frame = np.asanyarray(color_part.get_data())
        depth_frame = np.asanyarray(depth_part.get_data())
        depth_frame = depth_frame * self.depth_unit # mm -> m

        return state, color_frame, depth_frame

    def advance_tune(self, args_dict):
        # apply advance settings defined in yaml file
        for camera_type, settings_dict in args_dict.items():
            sensor = self.depth_sensor if camera_type == 'depth' else self.color_sensor

            for setting_name, setting_value in settings_dict.items():
                option = eval(f'rs.option.{setting_name}')

                if camera_type == 'depth' and setting_name == 'enable_auto_exposure':
                    ae_target_mode, ae_target_mi = setting_value['enable'], setting_value['mean_intensity']
                    sensor.set_option(option, ae_target_mode)
                    if ae_target_mode:
                        advnc_mode = rs.rs400_advanced_mode(self.profile.get_device())
                        aemi = advnc_mode.get_ae_control()
                        aemi.meanIntensitySetPoint = ae_target_mi
                        advnc_mode.set_ae_control(aemi)
                else:
                    sensor.set_option(option, setting_value)
        
        # waiting for modified settings to take effect
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def init_postprocess_filter(self, args_dict):
        # apply postprocess settings defined in yaml file
        for camera_type, filters_dict in args_dict.items():
            self.filters[camera_type] = self.filters.get(camera_type, [])
            if camera_type == 'color':
                self.filters[camera_type] = self.color_sensor.get_recommended_filters()
            else:
                for filter_name, filter_args in filters_dict.items():
                    self.filters[camera_type].append(eval(f'rs.{filter_name}(**{filter_args})'))
        
    def apply_filters_ls(self, data, filter_ls):
        for filter in filter_ls:
            data = filter.process(data)
        return data

class DataSource():
    def __init__(self, args):
        self.data_iter = self.init_data(args)
        
    def init_data(self, args):
        '''
        Transfer any img/video/camera data to an frame data iterator
        '''
        input = args.input # str or np.ndarray

        if isinstance(input,str):
            if input.isnumeric(): # camera
                return LoadRsCamera(args)
            elif os.path.exists(input): # dir, img, video
                return LoadImagesAndVideos(input)
            else: 
                raise ValueError(f"`input` must be a valid path or camera id(e.g. 0), but got {input}")
        
        elif isinstance(input,np.ndarray):
            return LoadPilAndNumpy(input)

        else:
            raise TypeError("Expected str or np.ndarray for argument `input`, but got {}".format(type(input)))

    def __len__(self):
        return len(self.data_iter)

    def __iter__(self):
        return self.data_iter.__iter__()

class Frame_Info(Results):
    """
    A class for storing and manipulating inference results. Inherit from the Results class.

    Attributes:
        depth_img (numpy.ndarray): Depth image as a numpy array.
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

        if_track (bool): Flag indicating whether tracking is enabled.
        depth (numpy.ndarray): Depth values as a numpy array. Unit meter. If data does not from camera, values are all 2000
        emergent_light_color (int): ascii code of closet traffic light color's abbraviation
        emergency_idx (list): each class's closet obj's index.
        fps (float): Frames can be processed per second.
        v0 (float): Velocity detected using track result and depth information, unit km/h.
        brake_a (float): Brake acceleration calculated by velocity, unit m/s^2.

        box_color (list): List of box colors for each class obj, negletable obj and emergent obj.
        hsv_color_range (dict): Dictionary of HSV color ranges for traffic light colors.
        valid_h (float): Valid top location, use to filt out traffic light boxes.
        valid_w (float): Valid horrizon range, use to filt out person boxes

    Attributes:
        emergency_info (ndarray): 

    Methods:
        perfect_box(): Assign boxes color according to box location, class and depth.
        get_light_color(roi): Get the color of the traffic light in the given region of interest (ROI).
    """
    def __init__(self, settings, depth_img, *args, **kwargs):
        super(Frame_Info, self).__init__(*args, **kwargs)
        self.depth_img = depth_img
        self.if_track = self.boxes.is_track if self.boxes is not None else False

        # get merge flag defined in yaml file
        self.if_merge = settings.get("merge_color_depth", False)
        self.merge_alpha = settings.get("merge_ratio", 0.5)

        # get box color defined in yaml file, or all boxes are green 
        self.box_color = settings.get("box_color", [[0, 255, 0]*len(self.names)]+[[192,192,192],[0, 0, 255]]) 
        self.box_color = [tuple(color) for color in self.box_color] # transfer to tuple, avoid plotting error
        
        # get traffic light color threshold defined in yaml file, or use default value
        self.hsv_color_range = settings.get("hsv_color_range", dict()) # hsv range of traffic light color
        if not self.hsv_color_range:
            self.hsv_color_range = {'Red1': ([0,43,46], [10,255,255]),    # hsv value:min,max
                                    'Red2': ([156,43,46], [180,255,255]),
                                    'Green': ([35,43,46], [89,255,255]),
                                    'Yellow': ([11,43,46], [34,255,255])}
        
        frame_h, frame_w = self.orig_shape
        h_margin = settings.get("h_margin", 0.013)
        w_margin = settings.get("w_margin", 0.1)
        self.valid_h = frame_h * h_margin # valid top location, use to filt out traffic light boxes
        self.valid_w = [frame_w * w_margin, frame_w * (1-w_margin)] # valid horrizon range, use to filt out person boxes

        self.emergent_light_color = ord("B")
        self.fps = np.nan
        self.v0 = np.nan
        self.brake_a = 0
        self.if_alert = False
    
    def __getitem__(self, idx):
        """(Rewrite)Just execute based on self.data, will not create a new obj"""
        return self.boxes.data[idx]

    def perfect_box(self): 
        '''Assign boxes color according to box location, class and depth'''
        num_boxes = len(self.boxes)
        if num_boxes == 0 or self.boxes.shape[-1] > 7: # no detections or already assigned color
            return
        
        # ndarray, (num_boxes, 9(8)), [x1, y1, x2, y2, (track_id), conf, cls, box_color_idx, traffic_color]
        boxes = np.c_[self.boxes.data.cpu().numpy(), 
                      np.array([np.nan]*num_boxes), # box color idx
                      np.array([np.nan]*num_boxes)] # traffic light color idx

        # get depth data
        det = boxes[:,:-2].copy() 
        self.depth = self.get_box_depth(det, self.depth_img) if self.depth_img is not None \
                     else np.array([2000]*num_boxes) # unit: meter
        
        # filt out neglect boxes
        center_x = (det[:,0]+det[:,2])/2
        grey_objs_idx = [idx for idx,(x,y) in enumerate(zip(center_x, det[:,3]))\
                            if x <= self.valid_w[0] or x >= self.valid_w[1] or y <= self.valid_h]
        boxes[grey_objs_idx, -2] = -2 # set box color to gray idx, which means this box is neglected
        
        # filt out emergency box
        frame_cls = set(det[:,-1])
        emergency_idx = []
        if 2000 in self.depth: # depth data invalid, filt by box loc 
            for cls in frame_cls:
                # pick the lowest box for people and zebra-crossing, but highest for traffic light
                pick_func = min if self.names[cls] == "T" else max 
                cls_idx = np.where(det[:,-1] == cls)[0]
                emergency_idx.append(pick_func(cls_idx, key=lambda x: det[x,3]))
        else: # depth data valid, filt by depth
            ascending_dis_idx = self.depth.argsort()
            for idx in ascending_dis_idx:
                if idx in grey_objs_idx: continue
                if len(frame_cls) == 0: break
                box_cls = det[idx,-1] 
                if box_cls in frame_cls:
                    emergency_idx.append(idx)
                    frame_cls.remove(box_cls)
        boxes[emergency_idx, -2] = -1 # set box color to red idx, which means this box is emergency

        # fill color and traffic light color information for each box
        for idx, box_data in enumerate(boxes):
            if box_data[-2] == -2: continue # skip neglectable boxes
            
            x1, y1, x2, y2 = map(int, box_data[:4])
            cls, box_color_idx = box_data[-3:-1]
            if np.isnan(box_color_idx):
                box_data[-2] = cls  
            else: # emergency box
                if self.names[cls] == "T":
                    self.emergent_light_color = self.get_light_color(self.orig_img[y1:y2, x1:x2, :].copy()) # traffic light color idx
                    box_data[-1] = self.emergent_light_color
        
        # update boxes data
        self.update(boxes=boxes)

    def get_box_depth(self, preds, depth_img):
        '''
        Return the average depth value of 3 points on the box's centerline as final depth value

        preds: tensor, (num_boxes,6), column meanings are [x1, y1, x2, y2, confidence, class]
        depth_imgs: ndarray, (H,W). Depth data(unit m)
        '''
        x1, y1, x2, y2 = preds[:,:4].T.astype(np.uint8) # x*,y* are vectors with shape (num_boxes,)
        x1 = np.clip(x1, 0, depth_img.shape[1]-1)
        x2 = np.clip(x2, 0, depth_img.shape[1]-1)
        y1 = np.clip(y1, 0, depth_img.shape[0]-1)
        y2 = np.clip(y2, 0, depth_img.shape[0]-1)
        
        top_points = y1, (x1+x2)//2
        mid_points = (y1+y2)//2, (x1+x2)//2
        buttom_points = y2, (x1+x2)//2
        
        depth_vals = np.c_[depth_img[top_points], depth_img[mid_points], depth_img[buttom_points]]
        return np.mean(depth_vals, axis=1)

    def get_light_color(self, roi:np.ndarray):
        '''
        Return the color of traffic light in ascii code, e.g. 82 for 'R'.
        roi: ndarray, (h,w,3). Box region of traffic light in origin image.
        '''
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_ths = cv2.threshold(roi_gray, 147, 255, cv2.THRESH_BINARY)
        ths_img = cv2.bitwise_and(roi,roi,mask=mask_ths)
        ths_img_hsv = cv2.cvtColor(ths_img, cv2.COLOR_BGR2HSV)
        
        possible_colors={}

        for color,(min_list,max_list) in self.hsv_color_range.items():
            mask = cv2.inRange(ths_img_hsv, np.array(min_list), np.array(max_list))
            #mask=cv2.medianBlur(mask, 3)   # erode the single pixel
            
            #mask=cv2.bitwise_and(roi,roi,mask=mask)
            #cv2.imshow('mask',mask)
            #cv2.waitKey(0)
            #cv2.destroyWindow('mask')

            mask = mask//255
            roi_ratio = np.sum(mask)/mask.size
            
            if 0.01 < roi_ratio < 0.33: # area limit
                possible_colors[roi_ratio]=color
            
        pred_color = possible_colors[max(possible_colors)][0] if possible_colors else 'B'  # get the first character of the color
        
        return ord(pred_color) # return color ascii

    def verbose(self):
        """Rewrite verbose method"""
        log_string = ""
        boxes = self.boxes.data

        if len(boxes) == 0:
            return f"(no detections), fps: {self.fps:.2f}"
        
        else:
            cls_count = Counter(boxes[:, -3])
            for cls_idx, cls_times in cls_count.items():
                log_string += f"{cls_times} {self.names[int(cls_idx)]}{'s' * (cls_times > 1)}, "
    
        log_string += f'{self.v0:.2f} km/h, ' if not np.isnan(self.v0) else 'None km/h, '
        log_string += f'{self.brake_a:.2f} m/s², ' if not np.isnan(self.brake_a) else '0.00 m/s², '
        log_string += f'fps: {self.fps:.2f}'

        return log_string

    def plot(self,
             line_width=None, font_size=None, font="Arial.ttf",
             pil=False, img=None, 
             labels=True,
             boxes=True,
             show=False,
             save_name=None):
        """
        (Rewrite)Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            show (bool): Whether to display the annotated image directly.
            save_name (str): Filename to save image.If it is None, will not save the annotated image.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        if img is None:
            img = self.orig_img

        names = self.names
        pred_boxes, show_boxes = self.boxes.data, boxes
        annotator = Annotator(
            deepcopy(self.cd_merge(img, self.depth_img) if self.if_merge and self.depth_img is not None else img),
            line_width, font_size, font,
            pil,  # Classify tasks default to pil=True
            example=names
        )

        # Plot Detect results
        if len(pred_boxes) != 0 and show_boxes:
            for box_id, box_data in enumerate(pred_boxes):
                cls_name, depth, box_color_idx = names[int(box_data[-3])], self.depth[box_id], int(box_data[-2])
                depth = f'{depth:.2f}' if depth < 2000 else 'None'
                
                label = (f' {cls_name} {depth} m' if cls_name!='T' else f' {cls_name} {chr(self.emergent_light_color)} {depth} m') \
                    if labels and box_color_idx == -1 else None # only show labels for emergency objs
                box = box_data[:4].squeeze() # xyxy
                annotator.box_label(box, label, color=self.box_color[box_color_idx])

                if box_color_idx == -1: # draw a red line from the center bottom of the image to the bottom of the box
                    img_shape = annotator.im.shape
                    x1,y1,x2,y2 = map(int, box)
                    cv2.line(annotator.im,
                             (int(img_shape[1]//2), img_shape[0]), 
                             (int((x1+x2)//2), y2), 
                             (0, 0, 255))

        # Show results
        if show:
            #annotator.show(self.path)
            cv2.imshow('doic', annotator.im)
            cv2.waitKey(1)  #(int(1000/self.fps))

        # Save results
        if save_name:
            annotator.save(save_name)

        return annotator.result()

    def cd_merge(self, c, d):
        '''
        c: color img
        d: depth img
        '''
        mask = cv2.normalize(d,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        mask = cv2.applyColorMap(mask,cv2.COLORMAP_JET)
        merge_img = cv2.addWeighted(c, 1-self.merge_alpha,
                                    mask, self.merge_alpha,
                                    0)
        return merge_img

    @property
    def emergency_info(self):
        if len(self.boxes) == 0: # no objs detected in this frame
            return np.zeros((0, 4))
        
        pick_col = [-3, 4, -1] if self.if_track else [-3, -3, -1] # cls, track id if track else cls, traffic light color
        # if tracking is disabled, use cls idx as track id
        # This mean accept the assumption that in a short time windows
        # each class's closest obj will not change. So we can use the distance of each class's closest obj to calculate v0

        return np.c_[self.boxes.data[:, pick_col], # cls, track_id, traffic light color
                     self.depth]              # depth

    @property
    def compact_data(self): # use for saving
        boxes_num = len(self.boxes)
        if not boxes_num: # no objs detected in this frame
            return self.orig_img, np.zeros(shape=(0, 14))
        
        return self.orig_img, np.c_[self.boxes.data, 
                                    self.depth,
                                    np.array([self.fps]*boxes_num),
                                    np.array([self.v0]*boxes_num),
                                    np.array([self.brake_a]*boxes_num),
                                    np.array([int(self.if_alert)]*boxes_num)]
