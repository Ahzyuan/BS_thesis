import os, torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.predictor import BasePredictor
from ultralytics.trackers.byte_tracker import BYTETracker

from utils import Frame_Info

class Callback_Base():
    def on_predict_start(self, *args, **kwargs):
        pass

    def on_predict_postprocess_end(self, results:list, *args, **kwargs):
        for res in results:
            res.perfect_box()
    
class Track_Callback(Callback_Base):
    '''Borrowed from ultralytics.tracker.track'''
    def __init__(self, args, model, persist=False):
        self.TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.cfg = args
        self.model = model # The predictor object to initialize trackers for. In this script, it is the instance of class Yolo8_Detracker
        self.persist = persist # whether to persist the trackers if they already exist
        
    def on_predict_start(self) -> None:
        """
        Initialize trackers for object tracking during prediction.
        """
        if hasattr(self.model, "trackers") and self.persist:
            return

        tracker_cls = self.TRACKER_MAP.get(self.cfg.tracker_type, BYTETracker)

        # only need one tracker if it is not stream input
        # for multi-streams: [tracker_cls(args=self.cfg, frame_rate=30) for _ in range(self.cfg.batch if model.stream else 1)]
        self.model.trackers = tracker_cls(args=self.cfg, frame_rate=30) 
    
    def on_predict_postprocess_end(self, im0s: list, results:list) -> None:
        """
        Postprocess detected boxes and update with object tracking.

        Args:
            im0s (list): List of color images.
            results (list): List of Frame_Info objs.
        """
        for i,img in enumerate(im0s):
            det = results[i].boxes.cpu().numpy() # det: (n,6) [x1, y1, x2, y2, conf, cls]
            if len(det) == 0: 
                results[i].perfect_box()
                continue
            
            tracks = self.model.trackers.update(det, img) # tracks: (n,8) [x1, y1, x2, y2, track_id, conf, cls, idx]
            if len(tracks) == 0: 
                results[i].perfect_box()
                continue

            idx = tracks[:, -1].astype(int)
            results[i].boxes.data = results[i][idx] # restore the detection sequence according to the box id

            update_args = {"boxes": torch.as_tensor(tracks[:, :-1])} # update boxes with track information
            results[i].update(**update_args) # boxes: (n,7) [x1, y1, x2, y2, track_id, conf, cls]
            
            results[i].perfect_box() # # update boxes with depth, fps, v0, a-, box_color, traffic_color

class Yolo8_Detracker(BasePredictor): # yolov8 detector or tracker
    def __init__(self, args):
        weight_name, weight_ext = os.path.splitext(os.path.basename(args.model))
        
        # Custom inference args
        custom = {
            "conf": getattr(args, "conf", 0.25),
            "iou": getattr(args, "conf", 0.45),
            "data": args.config, # this parse the calss name into the whole program
            "mode": "track" if args.if_track else "predict",
            "half": getattr(args, "half", True if 'half' in weight_name else False),
            "dnn": getattr(args, "dnn", True if 'onnx' in weight_ext else False)
            } 

        super(Yolo8_Detracker,self).__init__(overrides=custom) 
        self.setup_model(model=args.model, verbose=False)

        args.batch = self.args.batch # update batch size according to model args
        self.camera_flag = 1 if args.input.isnumeric() else 0 # used to receive depth information
        self.enable_depth= getattr(args, "enable_depth", False)
        self.det_analyse_settings = getattr(args, "det_analyse_settings", dict())
        self.det_analyse_settings["intri_mat"] = np.asanyarray(getattr(args, "cs_intrimat", \
                                                               self.det_analyse_settings.get("intri_mat", None))) # update intri_mat with realsense camera built-in value

        self.callback = Track_Callback(args, self, persist=True) if args.if_track else Callback_Base()
    
    def __call__(self, paths:list, imgs:list):
        # decode camera data, e.g. [(color_frame0,depth_frame0), (color_frame1,depth_frame1), ...]
        if self.camera_flag:
            color_imgs, depth_imgs = [],[]
            for frame_data in imgs:
                color_imgs.append(frame_data[0])
                depth_imgs.append(frame_data[1])
            imgs = color_imgs

        timer = ops.Profile(device=self.device) # record time in second
        
        self.callback.on_predict_start() # init tracker
        with timer:
            # Preprocess
            im = self.preprocess(imgs) # tensor, (B, 3, H, W)

            # Inference
            preds = self.model(im) # tensor (B,C,H,W) or [tensor (B,C,H,W)] when bs > 1

            # Postprocess
            results = self.postprocess(self.det_analyse_settings, # list of ndarray or Results objects
                                       paths, 
                                       preds,  
                                       im, 
                                       imgs, 
                                       depth_imgs=depth_imgs if self.camera_flag and self.enable_depth else None)
            self.callback.on_predict_postprocess_end(im0s=imgs, # update tracker
                                                     results=results) 
        
        # add process fps to frame_res
        duration = timer.dt / len(imgs) # batch mean process time in second
        fps = 1 / duration # batch mean fps
        for res in results:
            res.fps = fps            
        
        yield from results
    
    def postprocess(self, settings_dict, paths, preds, img, color_imgs, depth_imgs=None):
        """
        Post-processes predictions and returns a list of Results objects.
        Borrowed from `DetectionPredictor` in ultralytics/models/yolo/detect/predict.py

        settings_dict: dict, detection analyse settings defined in yaml file
        paths: image paths
        preds: inference result
        img: preprocess result
        color_imgs: color image list, each item is a ndarray with shape (H,W,3)
        depth_imgs: depth image list, each item is a ndarray with shape (H,W)
        """
        # `preds`: [torch.Tensor*B], each element has shape (num_boxes, 6 + num_masks). 
        # When `nc=0` in func `ops.non_max_suppression`, num_masks=0. So in this case, each element has shape (num_boxes, 6).
        # each column meaning is (x1, y1, x2, y2, confidence, class)
        preds = ops.non_max_suppression( 
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(color_imgs, list):  # input images are a torch.Tensor, not a list
            color_imgs = ops.convert_torch2numpy_batch(color_imgs)

        results = []
        for frame_idx, frame_pred in enumerate(preds): # frame_pred: tensor (num_boxes, 6). If no detected objs, will be (0,6)
            color_img = color_imgs[frame_idx]
            depth_img = depth_imgs[frame_idx] if depth_imgs else None
            
            frame_pred[:, :4] = ops.scale_boxes(img.shape[2:], frame_pred[:, :4], color_img.shape)
            
            res = Frame_Info(settings_dict,
                             depth_img=depth_img,
                             orig_img=color_img,
                             path=paths[frame_idx], 
                             names=self.model.names, 
                             boxes=frame_pred)
            
            results.append(res)
        
        return results

    def warmup(self):
        self.model.warmup(imgsz=(self.args.batch, 3, *self.imgsz))
