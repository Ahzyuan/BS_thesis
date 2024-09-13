import os
import numpy as np
from functools import partial

class Reactor():
    def __init__(self, args):
        self.win_size = getattr(args, "win_size", 3)
        self.max_a = 250 / (9*getattr(args, "min_zero2hundred_time", 4)) # unit m/s^2
        self.vt = getattr(args, "vt", 5) / 3.6 # unit m/s
        self.alert_thres = getattr(args, "alert_thres", 0.2)
        self.cls_names = getattr(args, "names", {i:str(i) for i in range(80)}) #args.names

        self.record_idx = 0
        self.frame_dis = {}  # key: obj_id   value: list[obj_dis] with length of win_size
        self.frame_fps = []  # list[float] with length of win_size
        self.frame_v0 = []   # list[float] with length of win_size, unit m/s

    def update(self, track_dis, fps):
        '''
        track_dis: ndarray, (num_boxes, 4), [cls, track_id, light_color, depth] if it is detected or it is an empty array
        '''
        # update each record obj's distance
        seen_obj = []
        for track_id, dis in track_dis[:, (1,3)]:
            if track_id not in seen_obj:
                seen_obj.append(track_id)
                self.frame_dis[track_id] = self.frame_dis.get(track_id, [2000]*self.record_idx) + [dis]
            else:
                self.frame_dis[track_id][-1] = min(dis, self.frame_dis[track_id][-1])
        for inactivate_obj in set(self.frame_dis.keys()) - set(track_dis[:, 1]):
            self.frame_dis[inactivate_obj].append(2000)

        # update fps
        self.frame_fps.append(fps)
        
        if self.record_idx == self.win_size-1:
            dis_mat = np.array(list(self.frame_dis.values())) # num_objs, win_size
            valid_objs = np.sum(dis_mat<2000, axis=-1) >= 2 # num_objs, discard objs whose have less than two valid dis
            valid_objs_idx = np.where(np.sum(dis_mat[:,1:]<2000, axis=-1) >= 1)[0] if dis_mat.any() else []
            dis_mat = dis_mat[valid_objs] # num_valid_objs, win_size
            
            v0 = self.get_v0(dis_mat) if len(dis_mat) else np.nan # unit: m/s

            if track_dis[:,-1].any():
                valid_light_idx = ~np.isnan(track_dis[:,-2])
                light_color_ascii = track_dis[valid_light_idx, -2][0].astype('int') if any(valid_light_idx) else ord('B')

                brake_a = self.get_brake_a(track_dis,  # unit: m/s^2
                                           v0, # v0
                                           self.vt,
                                           chr(light_color_ascii)) # light color

            else: # no detections in this frame
                brake_a = 0 # unit: m/s^2
            
            self.frame_v0.append(v0)
            if_alert, *beep_info = self.alert(brake_a) # beep_info: (duration, frequency)
            
            # discard the very first frame's data
            key_ls = list(self.frame_dis.keys())
            valid_key = [key_ls[key_idx] for key_idx in valid_objs_idx]
            self.frame_dis = {key: value[1:] for key, value in self.frame_dis.items() if key in valid_key}
            self.frame_fps.pop(0)
            if len(self.frame_v0) == self.win_size:
                self.frame_v0.pop(0)
        else:
            v0, brake_a = np.nan, 0
            if_alert = False
            beep_info = None
            self.record_idx += 1 # while not reach win_size, just keep storing the frame data

        return v0*3.6, brake_a, if_alert, beep_info  # unit: v0(km/h), brake_a(m/s^2), beep_info: (duration(second), frequency(Hz))

    def get_v0(self, dis_mat):

        def max_range_v0(dis_vector, fps_ls):
            '''Calculate v0 using the left most valid dis and right most valid dis'''
            valid_idx = np.where(dis_vector<2000)[0]
            l, h = valid_idx[0], valid_idx[-1]
            return (dis_vector[l] - dis_vector[h])/sum(fps_ls[l:h])
        
        # negative value means car is static, so v0=0; while positive value mean car is proceeding with this velocity
        obj_v0 = np.apply_along_axis(partial(max_range_v0, fps_ls=self.frame_fps), axis=-1, arr=dis_mat) # unit: m/s
        obj_v0[obj_v0<0] = 0
        
        v0 = obj_v0.mean()
        
        # judge if it is a valid v0 according to max accleration:
        if self.frame_v0:
            last_frame_v0 = self.frame_v0[-1]
            a = abs(v0 - last_frame_v0)/self.frame_fps[-1]
            v0 = v0 if a <= self.max_a else last_frame_v0+self.max_a*self.frame_fps[-1]*(v0-last_frame_v0)
            v0 = max(0,v0)

        v0 = 7.7-42*v0 if v0*420 > 7 else v0*420 # unit: km/h

        return v0 # unit: m/s, 60 is empirical scale
            
    def get_brake_a(self, track_dis, v0, vt, light_color):
        '''
        Calculate brake accleration according to v0, vt, light color and cls

        v0: current speed, unit m/s
        vt: target speed, unit m/s
        light_color: abbreviation of light color
        cls_idx: boxes class

        return: brake accleration, unit m/s^2
        '''
        if v0 != 0 or ~np.isnan(v0): 
            now_dis = track_dis[:,-1]
            min_idx = np.argmin(now_dis)
            now_min_dis = now_dis[min_idx]
            min_dis_cls = self.cls_names[track_dis[min_idx,0]].upper() 

            if now_min_dis < 2000:
                if min_dis_cls == 'Z':
                    if v0 <= vt:
                        return 0
                elif min_dis_cls =='T':
                    if light_color == 'G':
                        return 0
                    else:
                        vt=0
                else:  # min_dis_cls == 'P'
                    vt=0
            else:
                vt = v0
            
            brake_a = (vt+v0)*(vt-v0)/(2*now_min_dis)
        else:
            brake_a = 0

        return min(abs(brake_a), self.max_a)

    def alert(self, brake_a):
        degree = brake_a/self.max_a
        if_alert = degree > self.alert_thres 

        duration = max(5/self.frame_fps[-1], 0.1) if if_alert else 0 # second
        frequency = int(880 * (1 + degree)) if if_alert else 0 # Hz
        #os.system(f'play --no-show-progress --null --channels 1 synth {duration:.2f} sine {frequency}') # sudo apt install sox
        return if_alert, duration, frequency
