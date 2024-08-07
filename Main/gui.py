import os,cv2,torch
import numpy as np  
import tkinter as tk  
from sys import path
from time import sleep,time
from config import config
from tkinter import filedialog 
from tkinter import messagebox as mb
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image, ImageDraw, ImageFont
from ctypes import windll
from threading import Thread

from res_reactor_gui import res_reactor
from calibration_gui import camera_calibration, get_reproject_error, correct_img_size
from help_functions import get_focus_dis, get_cls_distance, get_light_color,\
                           attempt_load, letterbox, non_max_suppression, scale_coords

windll.shcore.SetProcessDpiAwareness(2)

# 优化数据存储

class TPZ_GUI():
    def __init__(self):
        ## All
        gui_info=config['gui_info']
        self.gui=tk.Tk()
        self.gui.title(gui_info['gui_title'])
        self.gui.geometry(gui_info['gui_w']+'x'+gui_info['gui_h'])  #1300,600
        # Global var
        self.after_id=0
        self.play_stop=0
        self.frame_source=tk.StringVar(value='realtime') # memory / realtime
        self.compress_grade=tk.IntVar(value=5)
        self.compress_params = [cv2.IMWRITE_PNG_COMPRESSION, self.compress_grade.get()]  # ratio: 0~9, larger for smaller img
        self.flip_cam=tk.IntVar(value=0)
        self.color_dict=gui_info['font_color']
        # Detect-related var
        self.img_size=tk.IntVar(value=608)
        self.conf=tk.Variable(value=0.25)
        self.iou=tk.Variable(value=0.45)
        self.w_margin=0.072 # use for classify ignorance objs
        self.zw_margin=0.1  # the margin for zebra crossing
        self.h_margin=0.013 # use for classify ignorance objs
        self.cam_id=[]
        #for cam_id in range(100):
        #    _=cv2.VideoCapture(cam_id)
        #    if _.isOpened():
        #        self.cam_id.append(cam_id)
        #    else:
        #        break
        self.cuda_device_num=torch.cuda.device_count()
        self.if_cpu=tk.IntVar(value=1) # default to use cpu for yolo
        self.cuda_var=[tk.IntVar(value=idx+1) for idx in range(self.cuda_device_num)]
        self.weight_path=tk.StringVar(value='')
        self.vt=tk.IntVar(value=30) # km/h
        self.detect_media_res={}
        self.detect_cam_res=[]
        self.replay_data=[]
        self.switch_source=0
        self.cam_detecting=0
        self.media_detecting=0
        # camera calibration-related var
        self.npy_path=tk.StringVar(value='')
        self.calib_img_path=tk.StringVar(value='')
        self.board_row=tk.IntVar(value=5)
        self.board_column=tk.IntVar(value=5)
        # saving-related var
        self.region_tips_str=tk.StringVar() # to show the frame slice
        self.video_save_path=tk.StringVar(value='')
        self.img_save_path=tk.StringVar(value='')
        self.save_size=tk.Variable(value=[1920,1080])
        self.save_fps=tk.IntVar(value=25)
        self.put_data=tk.IntVar(value=1) # to determine whether to show detected data on img
        # GUI-related var
        self.win_w=tk.IntVar(value=0)
        self.win_h=tk.IntVar(value=0)

        self.bar_val=tk.IntVar(value=0)
        self.last_bar_val=tk.IntVar(value=-1)
        self.media_close=tk.IntVar(value=1)
        self.cam_close=tk.IntVar(value=1)
        
        self.fps_val=tk.StringVar(value=['FPS:','0','frame/s']) 
        self.traffic_color=tk.StringVar(value='B') # 'R / G / B / Y 
        self.P_dis=tk.StringVar(value='None')
        self.Z_dis=tk.StringVar(value='None')
        self.T_dis=tk.StringVar(value='None')
        self.v_val=tk.StringVar(value='None')
        self.a_val=tk.StringVar(value='None')        

        self.frame_left =tk.Frame(self.gui)    # height=580, width=970
        self.frame_right=tk.Frame(self.gui)    # height=580, width=310
        self.frame_sep = tk.Frame(self.gui, width=2, bg='grey')  # height=500
        self.frame_left.pack(padx=5, pady=5, side='left')   #grid(padx=5, pady=5,row=0,column=0) #pack(padx=5, pady=5, side='left')
        self.frame_sep.pack(side='left')    #grid(pady=5,row=0,column=1)  #pack(side='left')
        self.frame_right.pack(padx=5, side='left', fill='x',expand=True)   #grid(padx=5, pady=5,row=0,column=2,sticky='nsew') #pack(padx=5, pady=5,side='left')
        
        menu_bar=tk.Menu(self.gui)
        menu_bar.add_command(label='设置', command=self.general_setting, background='blue')
        menu_bar.add_command(label='相机标定', command=self.calibration_setting, background='blue')
        self.gui.configure(menu=menu_bar)

        ## Left
        self.media_canvas=tk.Canvas(self.frame_left, bg='light gray')    # height=540, width=960
        self.bt_upload = tk.Button(self.media_canvas, command=self.upload_file, text='上传文件',cursor='hand2', height=1, width=9, activebackground='grey',bg='light grey',bd=1 ,relief='groove')
        self.bt_cam = tk.Button(self.media_canvas, command=self.open_cam, text='打开摄像头',cursor='hand2', height=1, width=9, activebackground='grey',bg='light grey',bd=1 ,relief='groove')      

        self.frame_left_down=tk.Frame(self.frame_left,bd=1,relief='groove')
        self.media_bar=tk.Scale(self.frame_left_down, from_=1, to=1, variable=self.bar_val ,takefocus=True, repeatdelay=10, repeatinterval=10, showvalue=False, relief='flat', sliderrelief='solid', troughcolor='light grey', activebackground='light green', bigincrement=5, cursor='hand2', orient=tk.HORIZONTAL)
        self.bar_info_front=tk.Label(self.frame_left_down, textvariable=self.bar_val)
        self.bar_info_back=tk.Label(self.frame_left_down)
        self.bt_close=tk.Button(self.frame_left_down, command=self.close_media, text='关闭',cursor='hand2', height=1,padx=10 , activebackground='red',bg='light grey',bd=1 ,relief='flat')
        self.bt_cam_close=tk.Button(self.frame_left, command=self.close_media, text='关 闭 摄 像 头', cursor='hand2', activebackground='red',bg='light grey',bd=1 ,relief='flat')

        self.media_canvas.pack()    #place(x=0,y=0)
        self.media_bar.pack(side='left')
        self.bar_info_front.pack(padx=3,side='left')
        self.bar_info_back.pack(side='left')
        self.bt_close.papackck(fill='both',expand=True)

        ## Right
        self.fps=tk.Label(self.frame_right, textvariable=self.fps_val, disabledforeground='light grey',state='disable')
        
        self.info_sep1=tk.Frame(self.frame_right, height=2, bg='light grey')
        self.info_sep2=tk.Frame(self.frame_right, height=2, bg='light grey')
        self.info_sep3=tk.Frame(self.frame_right, height=2, bg='light grey')
        
        self.P_pil=Image.open(path[0]+'\\source\\P.png')
        self.Z_pil=Image.open(path[0]+'\\source\\Z.png')
        self.T_B_pil=Image.open(path[0]+'\\source\\T_B.png')
        self.T_R_pil=Image.open(path[0]+'\\source\\T_R.png')
        self.T_G_pil=Image.open(path[0]+'\\source\\T_G.png')
        self.T_Y_pil=Image.open(path[0]+'\\source\\T_Y.png')

        self.P_icon=tk.Label(self.frame_right,image=ImageTk.PhotoImage(self.P_pil),anchor='e',state='disable')
        self.Z_icon=tk.Label(self.frame_right,image=ImageTk.PhotoImage(self.Z_pil),anchor='e',state='disable')
        self.T_icon=tk.Label(self.frame_right,image=ImageTk.PhotoImage(self.T_B_pil),anchor='e',state='disable')
        self.P_disfo=tk.Label(self.frame_right, textvariable=self.P_dis, disabledforeground='light grey',state='disable')
        self.Z_disfo=tk.Label(self.frame_right, textvariable=self.Z_dis, disabledforeground='light grey',state='disable')
        self.T_disfo=tk.Label(self.frame_right, textvariable=self.T_dis, disabledforeground='light grey',state='disable')
        self.P_unit=tk.Label(self.frame_right,text='m', disabledforeground='light grey',state='disable')
        self.Z_unit=tk.Label(self.frame_right,text='m', disabledforeground='light grey',state='disable')
        self.T_unit=tk.Label(self.frame_right,text='m', disabledforeground='light grey',state='disable')

        self.v_icon=tk.Label(self.frame_right,text='V :',anchor='center', disabledforeground='light grey',state='disable')
        self.v=tk.Label(self.frame_right, textvariable=self.v_val, disabledforeground='light grey',anchor='w',state='disable')
        self.a_icon=tk.Label(self.frame_right,text='aˉ:',anchor='center', disabledforeground='light grey',state='disable')
        self.a=tk.Label(self.frame_right, textvariable=self.a_val, disabledforeground='light grey',anchor='w',state='disable')
        self.v_unit=tk.Label(self.frame_right,text='m/s', disabledforeground='light grey',state='disable')
        self.a_unit=tk.Label(self.frame_right,text='m/s²', disabledforeground='light grey',state='disable')
        
        self.frame_right.grid_propagate(0)
        self.frame_right.rowconfigure((0,2,3,4,6,7),weight=1)
        self.frame_right.rowconfigure(9,weight=99)
        self.frame_right.columnconfigure(0,weight=2)
        self.frame_right.columnconfigure(1,weight=2)
        self.frame_right.columnconfigure(2,weight=1)

        self.fps.grid(row=0,column=0,columnspan=3,sticky='nesw')
        self.info_sep1.grid(row=1,column=0,columnspan=3,sticky='nesw')
        
        self.P_icon.grid(row=2,column=0,sticky='ew',pady=5)
        self.Z_icon.grid(row=3,column=0,sticky='ew')
        self.T_icon.grid(row=4,column=0,sticky='ew',pady=5)
        self.P_disfo.grid(row=2,column=1,sticky='e',pady=5)
        self.Z_disfo.grid(row=3,column=1,sticky='e')
        self.T_disfo.grid(row=4,column=1,sticky='e',pady=5)
        self.P_unit.grid(row=2,column=2,sticky='nesw',pady=5)
        self.Z_unit.grid(row=3,column=2,sticky='nesw')
        self.T_unit.grid(row=4,column=2,sticky='nesw',pady=5)
        self.info_sep2.grid(row=5,column=0,columnspan=3,sticky='nesw')

        self.v_icon.grid(row=6,column=0,sticky='e')
        self.a_icon.grid(row=7,column=0,sticky='e')
        self.v.grid(row=6,column=1,sticky='e')
        self.a.grid(row=7,column=1,sticky='e')
        self.v_unit.grid(row=6,column=2,sticky='nesw')
        self.a_unit.grid(row=7,column=2,sticky='nesw')
        self.info_sep3.grid(row=8,column=0,columnspan=3,sticky='nesw')

        self.frame_func=tk.Frame(self.frame_right)
        self.frame_func.grid(row=9,column=0,columnspan=3,sticky='snew')
        self.frame_func.rowconfigure(0,weight=2)
        self.frame_func.rowconfigure(1,weight=2)
        self.frame_func.rowconfigure(2,weight=2)
        self.frame_func.rowconfigure(3,weight=2)
        self.frame_func.columnconfigure(0,weight=29)
        self.frame_func.columnconfigure(1,weight=1)
        
        self.bt_play=tk.Button(self.frame_func, command=self.play, text='播\t放',state='disable', cursor='hand2', activebackground='red',disabledforeground='#B7B7B7',bg='light grey',bd=1 ,relief='flat')
        self.bt_detect=tk.Button(self.frame_func,command=self.detect_bt_func, text='开始检测',state='disable', cursor='hand2', activebackground='red',bg='light grey',bd=1 ,relief='flat',disabledforeground='#B7B7B7')
        self.bt_set_region=tk.Button(self.frame_func,command=self.set_region, text='(设置区间)',state='disable', cursor='hand2', activeforeground='red',bd=1,relief='flat',bg='#f0f0f0',disabledforeground='#B7B7B7')
        self.bt_save=tk.Button(self.frame_func, command=self.save, text='保存与导出', state='disable',cursor='hand2', activebackground='red',bg='light grey',bd=1 ,relief='flat',disabledforeground='#B7B7B7')
        self.bt_clear_detect=tk.Button(self.frame_func,text='清除检测结果',command=self.clear_detect,state='disable',cursor='hand2', activebackground='red',bg='light grey',bd=1 ,relief='flat',disabledforeground='#B7B7B7')

        self.bt_play.grid(row=0,column=0,columnspan=2, sticky='nesw',pady=5)
        self.bt_detect.grid(row=1,column=0,sticky='nesw',pady=5)
        self.bt_set_region.grid(row=1,column=1,sticky='nesw',pady=5)
        self.bt_save.grid(row=2,column=0,columnspan=2, sticky='nesw',pady=5)
        self.bt_clear_detect.grid(row=3,column=0,columnspan=2,sticky='news',pady=5)

        self.gui.bind('<Configure>', self.window_resize)

        self.gui.mainloop()

    def refresh_data(self, detect_data:list):
        '''
        detect_data: \n[(fps: int, color: str)\n, (P_dis, Z_dis, T_dis): float,\n (v_val, a_val): float]
        '''
        (fps, color), (P_dis, Z_dis, T_dis), (v_val, a_val)=detect_data
        
        self.fps_val.set(['FPS:',str(fps).zfill(2),'frame/s'])
        self.traffic_color.set(color)
        
        P_dis_str, Z_dis_str, T_dis_str=map(lambda x:'{:.2f}'.format(x), detect_data[1])

        v_val_str, a_val_str=map(lambda x:'{:.2f}'.format(x), detect_data[2])

        activate_widget=lambda x:x.configure(state='normal')
        disable_widget=lambda x:x.configure(state='disable')
        if fps>0:
            self.fps.configure(state='normal')
        else:
            self.fps.configure(state='disable')
        
        self.T_icon.configure(image=eval(f'self.T_{color}_pic'))
        self.T_disfo.configure(fg=self.color_dict[self.traffic_color.get()])
        self.T_unit.configure(fg=self.color_dict[self.traffic_color.get()])

        if P_dis<1000:
            self.P_dis.set(P_dis_str)
            list(map(activate_widget,(self.P_icon,self.P_disfo,self.P_unit)))
        else:
            self.P_dis.set('None')
            list(map(disable_widget,(self.P_icon,self.P_disfo,self.P_unit)))

        if Z_dis<1000:
            self.Z_dis.set(Z_dis_str)
            list(map(activate_widget,(self.Z_icon,self.Z_disfo,self.Z_unit)))
        else:
            self.Z_dis.set('None')
            list(map(disable_widget,(self.Z_icon,self.Z_disfo,self.Z_unit)))
        
        if T_dis<1000:
            self.T_dis.set(T_dis_str)
            list(map(activate_widget,(self.T_icon,self.T_disfo,self.T_unit)))
        else:
            self.T_dis.set('None')
            self.T_icon.configure(image=self.T_B_pic)
            list(map(disable_widget,(self.T_icon,self.T_disfo,self.T_unit)))
        
        if v_val<1000:
            self.v_val.set(v_val_str)
            list(map(activate_widget,(self.v_icon,self.v,self.v_unit)))
        else:
            self.v_val.set('None')
            list(map(disable_widget,(self.v_icon,self.v,self.v_unit)))

        if a_val<1000:
            self.a_val.set(a_val_str)
            list(map(activate_widget,(self.a_icon,self.a,self.a_unit)))
        else:
            self.a_val.set('None')
            list(map(disable_widget,(self.a_icon,self.a,self.a_unit)))

    def window_resize(self, event):
        new_w=self.gui.winfo_width()
        new_h=self.gui.winfo_height()
        now_w=self.win_w.get()
        now_h=self.win_h.get()
        
        if now_w!=new_w or now_h!=new_h:
            self.win_w.set(new_w)
            self.win_h.set(new_h)

            ## Left
            self.media_canvas.configure(height=new_h*0.9, width=new_w*0.74)
            self.canvas_width=self.media_canvas.winfo_reqwidth()
            self.canvas_height=self.media_canvas.winfo_reqheight()
            self.canvas_ratio=self.canvas_width/self.canvas_height

            self.frame_sep.configure(height=self.canvas_height)
            
            if self.media_close.get():
                self.bt_upload.place(x=self.canvas_width//2,y=self.canvas_height//2,anchor='s')
                self.bt_cam.place(x=self.canvas_width//2,y=self.canvas_height//2,anchor='n')
            else:
                self.media_canvas.delete('all')
                self.pick_img=ImageTk.PhotoImage(image=self.adaptive_size(self.pil_img))
                self.media_canvas.create_image(self.canvas_width//2, self.canvas_height//2, anchor='center', image=self.pick_img)
            
            self.media_bar.configure(length=self.canvas_width*0.8)

            ## Right
            self.frame_right.configure(height=self.canvas_height)

            sep_width=round(new_w*0.25)
            self.info_sep1.configure(width=sep_width)
            self.info_sep2.configure(width=sep_width)
            self.info_sep3.configure(width=sep_width)

            (
            self.P_adp,
            self.Z_adp,
            self.T_B_adp, self.T_R_adp, self.T_G_adp, self.T_Y_adp
            )=map(self.icon_adaptive_size,(
                self.P_pil,
                self.Z_pil,
                self.T_B_pil, self.T_R_pil, self.T_G_pil, self.T_Y_pil
            ))

            (
            self.P_pic,
            self.Z_pic,
            self.T_B_pic, self.T_R_pic, self.T_G_pic, self.T_Y_pic
            )=map(ImageTk.PhotoImage,(
                self.P_adp,
                self.Z_adp,
                self.T_B_adp, self.T_R_adp, self.T_G_adp, self.T_Y_adp
            ))
            self.P_icon.configure(image=self.P_pic) 
            self.Z_icon.configure(image=self.Z_pic) 
            self.T_icon.configure(image=eval(f'self.T_{self.traffic_color.get()}_pic')) 
            
            list(map(lambda x:x.configure(font=('微软雅黑',round(self.win_h.get()*0.015))),(
                # Left
                self.bt_upload,
                self.bt_cam,
                self.bt_close,
                self.bt_cam_close,
                self.bar_info_front,
                self.bar_info_back,
                # Right
                self.bt_play,
                self.bt_detect,
                self.bt_set_region,
                self.bt_save,
                self.bt_clear_detect
            )))

            list(map(lambda x:x.configure(font=('Consolas',round(self.win_h.get()*0.03))),(
                self.fps,
                self.P_disfo, self.Z_disfo, self.T_disfo,
                self.P_unit, self.Z_unit, self.T_unit,
                self.v_icon, self.a_icon,
                self.v, self.a,
                self.v_unit,self.a_unit
            )))

    def check_chinese(self, str):
        for chara in reversed(str):
            if u'\u4e00' <= chara <= u'\u9fff':
                return True
        return False

    def icon_adaptive_size(self, pil_img):
        origin_w,origin_h=pil_img.size
        origin_ratio=origin_w/origin_h

        target_h=round(self.win_h.get()*0.1) #if self.win_h.get()>1 else round(int(config['gui_info']['gui_h'])*0.1)
        target_w=round(origin_ratio*target_h)

        return pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def adaptive_size(self,img):
        if isinstance(img, Image.Image):
            origin_w,origin_h=img.size
        else:
            origin_h,origin_w=img.shape[:2]
        
        img_ratio=origin_w/origin_h

        if img_ratio > self.canvas_ratio:
            des_w = self.canvas_width
            des_h = self.canvas_width / img_ratio
        elif img_ratio < self.canvas_ratio:
            des_h = self.canvas_height
            des_w = self.canvas_height * img_ratio
        else:
            des_w = self.canvas_width
            des_h = self.canvas_height
        
        self.save_size.set([round(des_w), round(des_h)])
        if isinstance(img,Image.Image):
            return img.resize((round(des_w), round(des_h)), Image.Resampling.LANCZOS)
        else:
            return cv2.resize(img, (round(des_w), round(des_h)))

    def loading_win(self):
        self.load_win=tk.Toplevel(self.gui)
        self.load_win.title('文件载入中...')
        self.load_win.resizable(False,False)
        self.load_win.attributes('-topmost', 'true')

        main_frame=tk.Frame(self.load_win,width=100)
        main_frame.pack_propagate(0)
        main_frame.pack(padx=5,pady=5,fill='both',expand=True)

        main_frame.columnconfigure(0,weight=1)
        main_frame.columnconfigure(1,weight=1)
        main_frame.columnconfigure(2,weight=38)

        tk.Label(main_frame,text='进度:').grid(row=0,column=0,sticky='news')
        self.load_percent=tk.Label(main_frame,text='0%')
        self.load_percent.grid(row=0,column=1,sticky='news')
        self.load_bar=Progressbar(main_frame,value=0, maximum=self.count,orient=tk.HORIZONTAL, length=450)
        self.load_bar.grid(row=0,column=2,sticky='news',padx=5)
        self.loaded_num=tk.Label(main_frame,text='0')
        self.loaded_num.grid(row=0,column=3,sticky='e')
        tk.Label(main_frame,text='/ {}'.format(self.count)).grid(row=0,column=4,sticky='w')

        def affirm_behav():
            self.load_win.destroy()

        def cancel():
            self.cancel_load=1

        buttom_margin=tk.Frame(self.load_win,bg='#D0D0D0',height=40)
        buttom_margin.pack_propagate(0)
        buttom_margin.pack(fill='both',expand=True)
        tk.Button(buttom_margin,text='取消',command=cancel,bg='#D0D0D0', bd=1,relief='flat').pack(padx=10,side='right')
        self.bt_load_affirm=tk.Button(buttom_margin,text='确定',command=affirm_behav,state='disable',bg='#D0D0D0', bd=1,relief='flat')
        self.bt_load_affirm.pack(padx=10,side='right')

    def upload_file(self):
        self.media_close.set(0)

        video_type=('.mp4','.avi','.flv','.wmv')
        img_type=('.jpg','.png','.jpeg','.bmp','.webp','.tif','.tiff')
        self.files_path = list(filedialog.askopenfilenames(        # tuple, (abs_path1, abs_path2, ...)
            title='选择文件',
            filetypes=[('所有文件','.*'), ('图片文件',img_type), ('视频文件',video_type)]
        ))
        self.video_judge=[]
        self.cv_frame=[]
        self.video_fps=[]

        def load_from_memory():
            self.cancel_load=0
            now=1
            
            for item in self.files_path:
                file_type=os.path.splitext(item)[-1]
                if file_type in video_type:
                    cv_video=cv2.VideoCapture(item)
                    ret, cv_img =  cv_video.read() # res is on behalf of whether there are frames left
                    while ret:
                        if self.cancel_load:
                            self.cv_frame=[]
                            self.load_win.destroy()
                            break
                        self.load_percent.configure(text=str(int(now*100/self.count))+'%')
                        self.load_bar.configure(value=now)
                        self.loaded_num.configure(text=str(now))
                        self.cv_frame.append(np.array(cv2.imencode('.png',cv_img,self.compress_params)[1]).tobytes())
                        ret, cv_img =  cv_video.read()
                        now+=1
                    cv_video.release()
                else:
                    if self.cancel_load:
                        self.cv_frame=[]
                        self.load_win.destroy()
                        break

                    cv_img=cv2.imread(item)
                    self.cv_frame.append(np.array(cv2.imencode('.png',cv_img,self.compress_params)[1]).tobytes())
                    self.load_percent.configure(text=str(int(now*100/self.count))+'%')
                    self.load_bar.configure(value=now)
                    self.loaded_num.configure(text=str(now))
                    now+=1
            
            if not self.cancel_load:
                self.bt_load_affirm.configure(state='normal')

                self.frame_left_down.pack_forget()
                self.frame_left_down.pack(fill='x',expand=True)
                self.region_info=[1,self.count]  # set the default regio for exporting
                self.region_tips_str.set(f'{self.region_info}')
                self.bt_play.configure(state='normal')
                self.bt_detect.configure(state='normal')
                self.bt_set_region.configure(state='normal')
                self.bt_save.configure(state='normal')
                self.media_bar.configure(to=self.count)
                self.bar_info_back.configure(text='/ {}'.format(self.count))

                self.show_img() 
            else:
                self.cv_frame=[]
                self.close_media()               

        if self.files_path:
            self.count=0
            error_files=[]

            for item in self.files_path:
                if self.check_chinese(item):
                    error_files.append(item)
                else:
                    file_type=os.path.splitext(item)[-1]
                    if file_type in video_type:
                        cv_video=cv2.VideoCapture(item)
                        video_start=self.count+1
                        self.count+=cv_video.get(cv2.CAP_PROP_FRAME_COUNT)
                        self.video_fps.append(cv_video.get(cv2.CAP_PROP_FPS))
                        self.video_judge.append(lambda x,start=video_start,end=self.count: (start<= x <=end, item, start))
                        cv_video.release()

                    elif file_type in img_type:
                        self.count+=1
                    
                    else:
                        error_files.append(item)
            
            if error_files:
                self.files_path=list(set(self.files_path)-set(error_files))
                error_str='\n'.join(error_files)
                mb.showerror(title='文件错误', message=f'以下文件路径含有中文，或扩展名错误:\n\n{error_str} \n\n仅支持以下格式:\n视频: {video_type}\n图片: {img_type}')
            
            if self.files_path:
                self.count=int(self.count)

                if self.frame_source.get()=='memory':
                    self.loading_win()
                    Thread(target=load_from_memory,args=()).start()
                    self.bt_upload.place_forget()
                    self.bt_cam.place_forget()

                else:
                    self.frame_left_down.pack_forget()
                    self.frame_left_down.pack(fill='x',expand=True)

                    self.region_info=[1,self.count]  # set the default region for exporting
                    self.region_tips_str.set(f'{self.region_info}')

                    self.bt_play.configure(state='normal')
                    self.bt_detect.configure(state='normal')
                    self.bt_set_region.configure(state='normal')
                    self.bt_save.configure(state='normal')

                    self.media_bar.configure(to=self.count)
                    self.bar_info_back.configure(text='/ {}'.format(self.count))
                    
                    self.bt_upload.place_forget()
                    self.bt_cam.place_forget()
                    self.show_img()
        else:
            self.close_media()

    def open_cam(self):
        self.media_close.set(0)
        self.cam_close.set(0)
        self.bt_upload.place_forget()
        self.bt_cam.place_forget()
        self.bt_cam_close.pack(fill='x', pady=2)
        self.bt_detect.configure(state='normal')

        cam=cv2.VideoCapture(0)
        self.cam_fps=cam.get(cv2.CAP_PROP_FPS)
        while 1:
            if self.cam_close.get():
                self.bt_cam_close.pack_forget()
                break

            _, cv_img = cam.read() 
            if self.flip_cam.get():
                cv_img=cv2.flip(cv_img, 1)
            self.res_cv_image=self.adaptive_size(cv_img) # cv_img
            
            if not self.cam_detecting:
                self.pil_img=Image.fromarray(cv2.cvtColor(self.res_cv_image, cv2.COLOR_BGR2RGB))
                self.pick_img=ImageTk.PhotoImage(image=self.pil_img)
                self.media_canvas.create_image(self.canvas_width//2, self.canvas_height//2, anchor='center', image=self.pick_img)
            
            self.gui.update()

        cam.release()

    def get_new_frame(self,now_loc):
        video_flag=0
        refresh_data=[(0,'B'),(2000,2000,2000),(2000,2000)]

        if now_loc in self.detect_media_res:
            self.res_cv_image=self.adaptive_size(self.detect_media_res[now_loc][0])
            refresh_data=self.detect_media_res[now_loc][1]
        
        else:
            if self.frame_source.get()=='memory':
                if not self.replay_data:
                    self.res_cv_image=self.adaptive_size(cv2.imdecode(np.frombuffer(self.cv_frame[self.bar_val.get()-1], np.uint8),flags=1))
                else:
                    self.res_cv_image=self.adaptive_size(self.replay_data[self.bar_val.get()-1][0])
                    refresh_data=self.replay_data[self.bar_val.get()-1][1]
            else:
                if now_loc!=self.last_bar_val.get():
                    for video in self.video_judge:
                        res=video(now_loc)
                        if res[0]:
                            self.file_path=res[1]
                            self.img_idx=now_loc-res[-1]
                            video_flag=1
                            break
                    else:
                        self.img_idx=now_loc
                        self.file_path=self.files_path[self.img_idx-1]

                    if video_flag==1:
                        cv_video=cv2.VideoCapture(self.file_path)
                        cv_video.set(cv2.CAP_PROP_POS_FRAMES, self.img_idx)
                        _, cv_img = cv_video.read()
                        cv_video.release()
                    else:
                        cv_img=cv2.imread(self.file_path)
                    self.res_cv_image=self.adaptive_size(cv_img)

        return refresh_data
        
    def show_img(self):
        now_loc=self.bar_val.get()

        if now_loc!=self.last_bar_val.get():
            refresh_data=self.get_new_frame(now_loc)
            self.pil_img=Image.fromarray(cv2.cvtColor(self.res_cv_image, cv2.COLOR_BGR2RGB))
            self.pick_img=ImageTk.PhotoImage(image=self.pil_img)
            self.media_canvas.create_image(self.canvas_width//2, self.canvas_height//2, anchor='center', image=self.pick_img)
            if refresh_data:
                self.refresh_data(refresh_data)

            self.last_bar_val.set(self.bar_val.get())

        self.after_id=self.gui.after(10,self.show_img)

    def close_media(self):
        self.media_close.set(1)
        self.cam_close.set(1)
        self.bar_val.set(1)
        self.last_bar_val.set(0)
        if self.after_id:
            self.gui.after_cancel(self.after_id)
        self.video_judge=[]
        self.video_fps=[]
        self.cv_frame=[]
        self.replay_data=[]
        self.detect_cam_res=[]
        self.detect_media_res={}
        self.yolo_model=None
        self.reactor=None
        if self.switch_source:
            self.frame_source.set('realtime')

        self.media_canvas.delete('all')
        self.bt_upload.place(x=self.canvas_width//2,y=self.canvas_height//2,anchor='s')
        self.bt_cam.place(x=self.canvas_width//2,y=self.canvas_height//2,anchor='n')
        self.frame_left_down.pack_forget()

        self.fps_val.set(['FPS:','0','frame/s'])
        self.traffic_color.set('B')
        self.P_dis.set([None])
        self.Z_dis.set([None])
        self.T_dis.set([None])
        self.v_val.set([None])
        self.a_val.set([None])
        
        disable_widget=lambda x:x.configure(state='disable')
        list(map(disable_widget,(
            self.fps,
            self.P_icon, self.Z_icon, self.T_icon,
            self.P_disfo, self.Z_disfo, self.T_disfo,
            self.P_unit, self.Z_unit, self.T_unit,
            self.v_icon, self.a_icon,
            self.v, self.a,
            self.v_unit, self.a_unit,
            self.bt_play,
            self.bt_detect,
            self.bt_set_region,
            self.bt_save,
            self.bt_clear_detect
        )))
        self.T_icon.configure(image=self.T_B_pic)
        self.bt_play.configure(text='播\t放',command=self.play,state='disabled')
        self.bt_detect.configure(text='开始检测', command=self.detect_bt_func)

    def play(self):
        now_loc=self.bar_val.get()

        def stop():
            self.play_stop=1

        self.bt_play.configure(text='暂停播放',command=stop)

        while now_loc<=self.count:
            if self.play_stop:
                self.play_stop=0
                self.bt_play.configure(text='播\t放',command=self.play)
                break
            self.bar_val.set(now_loc)
            now_loc+=1
            self.media_canvas.update()
            if self.replay_data:
                sleep(0.05) 
            else:
                sleep(0.01)
        else:
            self.play_stop=0
            self.bt_play.configure(text='播\t放',command=self.play)    

    def replay_cam(self):
        origin_frame_source=self.frame_source.get()
        replay_data=self.detect_cam_res
        self.close_media()
        self.replay_data=replay_data

        self.count=len(self.replay_data)
        self.region_info=[1,self.count]  # set the default region for exporting
        self.region_tips_str.set(f'{self.region_info}')
        self.frame_source.set('memory')

        self.media_close.set(0)
        self.cam_close.set(1)
        self.frame_left_down.pack_forget()
        self.frame_left_down.pack(fill='x',expand=True)
        self.media_bar.configure(to=self.count)
        self.bar_val.set(1)
        self.bar_info_back.configure(text='/ {}'.format(self.count))

        self.bt_play.configure(state='normal',text='播\t放',command=self.play)
        self.bt_detect.configure(state='disable')
        self.bt_set_region.configure(state='normal')
        self.bt_save.configure(state='normal')
                
        self.bt_upload.place_forget()
        self.bt_cam.place_forget()
        self.show_img()
        if origin_frame_source!='memory':
            self.switch_source=1

    def set_region(self):
        if self.count==1:
            mb.showerror(title='帧数量过少',message='目前已上传 1 帧\n\n无法对1帧数据设置区间！')
            return 
        
        self.region_tips_str.set(f'{self.region_info}')

        def set_start():
            start=self.bar_val.get()
            end=self.region_info[1]
            if start >= end:
                mb.showerror(title='值错误',message=f'你设置的开始值：{start}\n结束值：{end}\n开始值必须小于结束值\n请重新设置开始值!')
            else:
                self.region_info[0]=start
                lb_start.configure(textvariable=start,state='disable',bg='#f0f0f0',relief='ridge')
                bt_start.configure(state='disable',relief='ridge')
                bt_affirm.configure(state='normal')

            self.region_tips_str.set(f'{self.region_info}')

        def set_end():
            end=self.bar_val.get()
            start=self.region_info[0]
            if start >= end:
                mb.showerror(title='值错误',message=f'开始值：{start}\n你设置的结束值：{end}\n开始值必须小于结束值\n请重新设置结束值!')
            else:
                self.region_info[1]=end
                lb_end.configure(textvariable=end,state='disable',bg='#f0f0f0',relief='ridge')
                bt_end.configure(state='disable',relief='ridge')
                bt_affirm.configure(state='normal')

            self.region_tips_str.set(f'{self.region_info}')

        set_win=tk.Toplevel(self.gui)
        set_win.title('设置检测区间')
        set_win.resizable(False,False)
        set_win.attributes('-topmost', 'true')

        main_frame=tk.Frame(set_win)
        main_frame.pack_propagate(0)
        main_frame.pack(padx=5,pady=5,fill='both',expand=True)

        main_frame.columnconfigure(0,weight=1)
        main_frame.columnconfigure(1,weight=47)
        main_frame.columnconfigure(2,weight=1)
        main_frame.columnconfigure(3,weight=1)

        tk.Label(main_frame,text="请移动滑块，设置将要处理的帧区间：",anchor='center').\
            grid(row=0,column=0,columnspan=4,sticky='s')
        
        tk.Label(main_frame,text='开始(含):').grid(pady=5,row=1,column=0,sticky='w')
        lb_start=tk.Label(main_frame, textvariable=self.bar_val,bg='white',anchor='w',bd=2,relief='groove')
        lb_start.grid(padx=5,pady=5,row=1,column=1,sticky='ew')
        bt_start=tk.Button(main_frame,text='设置',command=set_start,bd=2,relief='solid')
        bt_start.grid(pady=5,row=1,column=3,sticky='nsew')
        
        tk.Label(main_frame,text='结束(含):').grid(pady=5,row=2,column=0,sticky='w')
        lb_end=tk.Label(main_frame, textvariable=self.bar_val,bg='white',anchor='w',bd=2,relief='groove')
        lb_end.grid(padx=5,pady=5,row=2,column=1,sticky='ew')
        bt_end=tk.Button(main_frame,text='设置',command=set_end,bd=2,relief='solid')
        bt_end.grid(pady=5,row=2,column=3,sticky='ew')

        info_frame=tk.Frame(main_frame)
        info_frame.grid(row=3,column=0,columnspan=4,sticky='ew',pady=5)
        tk.Label(info_frame,text='将要处理的帧区间:',fg='red').pack(side='left')
        tk.Label(info_frame,textvariable=self.region_tips_str,foreground='red').pack(side='left')

        tk.Label(main_frame,text="提示:",\
                 fg='#808080',anchor='w').grid(row=4,column=0,columnspan=4,sticky='w')
        tk.Label(main_frame,text='精确定位: 在主窗口使用Tab选择滑动条，使用 “方向键” 进行精确定位',\
                 fg='#808080',anchor='w').grid(row=5,column=0,columnspan=4,sticky='w')
        tk.Label(main_frame,text='快速定位: 在主窗口使用Tab选择滑动条，使用 “ctrl+方向键” 进行快速定位；或直接拖动滑块',\
                 fg='#808080',anchor='w').grid(row=6,column=0,columnspan=4,sticky='w')

        def cancel():
            set_win.destroy()

        def affirm_behav():
            self.bar_val.set(self.region_info[0])
            set_win.destroy()
        
        def reset():
            self.region_info=[1,self.count]
            self.region_tips_str.set(f'{self.region_info}')
            lb_start.configure(state='normal',bg='white',textvariable=self.bar_val,relief='groove')
            bt_start.configure(state='normal',relief='solid')
            lb_end.configure(state='normal',bg='white',textvariable=self.bar_val,relief='groove')
            bt_end.configure(state='normal',relief='solid')

        buttom_margin=tk.Frame(set_win,bg='#D0D0D0',height=40)
        buttom_margin.pack_propagate(0)
        buttom_margin.pack(fill='both',expand=True)
        tk.Button(buttom_margin,text='取消',command=cancel,bg='#D0D0D0', bd=1,relief='flat').pack(padx=10,side='right')
        bt_clsset=tk.Button(buttom_margin,text='重新设置',command=reset,bg='#D0D0D0', bd=1,relief='flat')
        bt_clsset.pack(padx=10,side='left')
        bt_affirm=tk.Button(buttom_margin,text='确定',command=affirm_behav,state='disabled',bg='#D0D0D0', bd=1,relief='flat')
        bt_affirm.pack(padx=10,side='right')
    
    def draw_box(self, im0, cls_list, objs):
        colors = [[151, 46, 240],[240, 174, 46],[46,124,240],[192,192,192]]   #T:purple Z:orange P:blue Invalid: grey
        if isinstance(objs,dict):
            objs=np.r_[list(objs.values())]
        
        for obj in objs:    # obj:x_l, y_l, x_r, y_r, conf, cls, (cen_x, y_r, dis)
            xl, yl, xr, yr, _, cls_idx, *label_data=obj
            box_color=colors[int(cls_idx)]
            c1, c2 = (int(xl), int(yl)), (int(xr), int(yr)) 
            if label_data:
                cv2.rectangle(im0, c1, c2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA) 
                cv2.line(im0,(im0.shape[1]//2, im0.shape[0]) , (round(label_data[0]), round(label_data[1])), (0, 0, 255))
                label_str=cls_list[int(cls_idx)]+' {:.1f} m'.format(label_data[-1])
                t_size = cv2.getTextSize(label_str, 0, fontScale=1 / 3, thickness=1)[0]   #(fw,fh)
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(im0, c1, c2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, label_str, (c1[0], c1[1] - 2), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(im0, c1, c2, box_color, thickness=1, lineType=cv2.LINE_AA)
        return im0

    def detect_detail(self, obj_names, reactor, video_fps, bar_val=0):
        t1 = time()

        im0=self.res_cv_image
        img = letterbox(im0, self.img_size.get(), stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = self.yolo_model(img, augment=False)[0]
        det = non_max_suppression(pred, self.conf.get(), self.iou.get(), classes=None, agnostic=False)[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            tar_obj_dict, grey_objs, else_objs=get_focus_dis(det, obj_names, self.valid_h, self.valid_w, self.valid_zw)
            tar_obj_dict=get_cls_distance(im0.copy(), tar_obj_dict, self.intri_mat)

            cls_dis={'T':2000,'Z':2000,'P':2000}
            for cls in tar_obj_dict:
                cls_dis[cls]=tar_obj_dict[cls][-1]
            if not reactor.dis_mat.any():
                reactor.dis_mat=np.array(list(cls_dis.values())).reshape(-1,1)
            else:
                reactor.dis_mat=np.append(reactor.dis_mat, np.array(list(cls_dis.values())).reshape(-1,1), axis=1)

            if 'T' in tar_obj_dict:
                T_xmin,T_ymin,T_xmax,T_ymax=map(int,tar_obj_dict['T'][:4])
                color=get_light_color(im0.copy()[T_ymin:T_ymax,T_xmin:T_xmax])
            else:
                color='B'

            im0=self.draw_box(im0,obj_names, tar_obj_dict)
            im0=self.draw_box(im0,obj_names, else_objs)
            self.res_cv_image=self.draw_box(im0,obj_names, grey_objs)
            
            v0=2000
            brake_a=2000
            if video_fps:
                dis_collect_num=reactor.dis_mat.shape[-1]
                if dis_collect_num>=2:
                    v0=reactor.get_v0(video_fps)
                    brake_a=reactor.get_brake_a(v0, vt=self.vt.get()*5/18, light_color=color, cls_names=obj_names)

                    if dis_collect_num==3:
                        reactor.dis_mat=np.delete(reactor.dis_mat, 0, axis=1)               
            
            t2 = time()
            fps = round(1/(t2 - t1))
            res_data=[(fps, color), (cls_dis['P'], cls_dis['Z'], cls_dis['T']), (v0,brake_a)]
            if bar_val:
                self.detect_media_res[bar_val]=(self.res_cv_image, res_data)
        else:
            t2 = time()
            fps = round(1/(t2 - t1))
            res_data=[(fps, 'B'), (2000, 2000, 2000), (2000,2000)]
        
        self.pil_img=Image.fromarray(cv2.cvtColor(self.res_cv_image, cv2.COLOR_BGR2RGB))
        self.pick_img=ImageTk.PhotoImage(image=self.pil_img)
        self.media_canvas.create_image(self.canvas_width//2, self.canvas_height//2, anchor='center', image=self.pick_img)
        self.refresh_data(res_data)
        
        if not bar_val:
            self.detect_cam_res.append((self.res_cv_image, res_data))

    def detect_bt_func(self):
        if not os.path.exists(self.weight_path.get()):
            mb.showerror(title='YoLoV5模型不存在',message='请前往  菜单栏--设置--YoLoV5模型路径  重新指定路径')
            return
        if not os.path.exists(self.npy_path.get()):
            mb.showerror(title='标定文件不存在',message='请前往  菜单栏--设置--相机内参  重新指定路径\n若未进行相机标定, 请前往  菜单栏--相机标定  上传标定图片，并保存内参文件')
            return         
            
        def stop_detect():
            self.reactor=None
            if self.media_detecting:
                self.media_detecting=0
                self.bt_play.configure(state='normal')
                self.bt_close.configure(state='normal')
                self.bt_set_region.configure(state='normal')
                self.bt_save.configure(state='normal')
                self.bt_detect.configure(text='开始检测', command=self.detect_bt_func)
                self.bt_clear_detect.configure(state='normal')
                if self.after_id:
                    self.gui.after_cancel(self.after_id)
                self.show_img()
            else:
                self.cam_detecting=0
                if self.after_id:
                    self.gui.after_cancel(self.after_id)
                self.bt_play.configure(state='normal',text='回\t放',command=self.replay_cam)
                self.bt_cam_close.configure(state='normal')
                self.bt_detect.configure(text='开始检测', command=self.detect_bt_func)
                self.bt_clear_detect.configure(state='normal')
                self.refresh_data([(0,'B'),(2000,2000,2000),(2000,2000)])

        def media_detect():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if self.detect_loc<=self.region_info[1]:
                self.bar_val.set(self.detect_loc)
                _ = self.get_new_frame(self.detect_loc)
                try:
                    video_fps=self.video_fps[list(map(lambda x:x(self.detect_loc)[0],self.video_judge)).index(True)]
                except:
                    video_fps=0
                
                self.detect_detail(self.obj_names,self.reactor,video_fps,bar_val=self.detect_loc)
                self.detect_loc+=1
                self.after_id=self.gui.after(10,media_detect)
            else:
                self.gui.after_cancel(self.after_id)
                stop_detect()
        
        def cam_detect():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.detect_detail(self.obj_names,self.reactor,self.cam_fps)
            self.after_id=self.gui.after(int(1000/self.cam_fps),cam_detect)
        
        if self.if_cpu.get():
            self.device='cpu'
        else:
            choose_gpu=np.array(list(map(lambda x:x.get()!=0,self.cuda_var)))
            env_idx=','.join(map(str,np.where(choose_gpu==True)[0].tolist()))
            os.environ['CUDA_VISIBLE_DEVICES']= env_idx
            self.device='cuda:'+','.join(list(map(str,list(range(sum(choose_gpu))))))

        self.yolo_model=attempt_load(self.weight_path.get(), map_location=self.device)
        self.obj_names = self.yolo_model.module.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names
        self.reactor=res_reactor()

        self.bt_play.configure(state='disable')
        self.bt_save.configure(state='disable')
        self.bt_set_region.configure(state='disable')
        self.bt_detect.configure(text='暂停检测', command=stop_detect)

        frame_shape=self.res_cv_image.shape
        #self.frame_buttom_center=(frame_shape[0],frame_shape[1]//2)
        self.valid_h=frame_shape[0]*self.h_margin
        self.valid_w=[frame_shape[1]*self.w_margin, frame_shape[1]*(1-self.w_margin)]
        self.valid_zw=[frame_shape[1]*self.zw_margin, frame_shape[1]*(1-self.zw_margin)]

        if self.cam_close.get():
            self.media_detecting=1
            self.gui.after_cancel(self.after_id)
            self.detect_loc=self.region_info[0]
            self.last_bar_val.set(self.region_info[0]-1)    # to successfully get the first frame in the func self.get_new_frame()

            self.bt_close.configure(state='disable')
            media_detect()
        else:
            self.cam_detecting=1
            self.bt_cam_close.configure(state='disable')
            cam_detect()
        
    def clear_detect(self):
        self.detect_media_res={}
        self.detect_cam_res=[]
        self.yolo_model=None
        self.bt_clear_detect.configure(state='disable')

        if self.cam_close.get():
            self.last_bar_val.set(self.bar_val.get()-1) # to repaint the present frame
        
        mb.showinfo(title='提醒',message='已成功清除！')
                    
    def save(self):   
        img_type=('.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff')
        video_type=('.mp4','.avi','.wmv')

        def delete_img_path():
            self.img_save_path.set('')
            bt_save_img.configure(state='disable')
            if not self.video_save_path.get():
                bt_opendir.configure(state='disabled')

        def choose_img_dir():
            choose_path=filedialog.asksaveasfilename(
                title='选择保存位置',
                defaultextension='.png',
                initialfile='frame',
                filetypes=[('所有图片',('.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff')),
                           ('可移植网络图形',('.png')),
                           ('JPG文件交换格式',('.jpg','.jpeg')),
                           ('Windows 位图',('.bmp')),
                           ('WEBP文件',('.webp')),
                           ('Tag 图像文件格式',('.tif','.tiff'))]
            )
            if not choose_path:
                self.img_save_path.set('')
                bt_save_img.configure(state='disable')
            else:
                self.img_save_path.set(choose_path)
                if not self.check_chinese(choose_path):
                    bt_save_img.configure(state='normal')
                else:
                    mb.showerror(title='文件错误',message='所选路径不可包含中文\n请重新选择!')
                    delete_img_path()
            img_save_bar.configure(value=0)

        def delete_video_path():
            self.video_save_path.set('')
            fps_frame.grid_forget()
            bt_save_video.configure(state='disable')
            if not self.img_save_path.get():
                bt_opendir.configure(state='disabled')

        def choose_video_dir():
            choose_path=filedialog.asksaveasfilename(
                title='选择保存位置',
                defaultextension='.mp4',
                initialfile='detect_video',
                filetypes=[('所有视频',('.mp4','.avi','.flv')),
                           ('MPEG4 媒体',('.mp4')),
                           ('Winodws 媒体',('.avi','.wmv'))]
            )
            if not choose_path:
                fps_frame.grid_forget()
                bt_save_video.configure(state='disable')
                return 
            else:
                self.video_save_path.set(choose_path)
                if not self.check_chinese(choose_path):
                    fps_frame.grid(row=3,column=0,sticky='news')
                    bt_save_video.configure(state='normal')
                else:
                    mb.showerror(title='文件错误',message='所选路径不可包含中文\n请重新选择！')
                    delete_video_path()
                try:
                    if choose_path in self.files_path:
                        mb.showerror(title='文件错误',message='不可保存为当前打开的文件\n请对保存文件进行重命名，或保存为其他格式！')
                        delete_video_path()
                except:
                    pass
            video_save_bar.configure(value=0)

        def open_dir():
            img_save_path=os.path.split(self.img_save_path.get())[0]
            video_save_path=os.path.split(self.video_save_path.get())[0]
            for path in [img_save_path,video_save_path]:
                if path:
                    os.system(f'start {path}')

        def save_detect_data(cv_img):
            Tcolor_dict={
                'R':'#FF3C3C',
                'G':'#00FF80',
                'B':'#FFFFFF',
                'Y':'#FFF206'
            }
            frame_h, frame_w=cv_img.shape[:2]
            pil_img=Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_drawing = ImageDraw.Draw(pil_img)

            fps_str='FPS '+self.fps_val.get().split(', ')[1].strip("'")
            sep_str='-'*12
            P_str='P '+self.P_dis.get()+' m'
            Z_str='Z '+self.Z_dis.get()+' m'
            T_str='T '+self.T_dis.get()+' m'
            v_str='V  '+self.v_val.get()+' m/s'
            a_val='aˉ '+self.a_val.get()+' m/s²'
            T_color=Tcolor_dict[self.traffic_color.get()]
                
            log_str_up='\n'.join([fps_str, sep_str, v_str, a_val, sep_str, P_str, Z_str])
            log_str_down=T_str
            
            text_size=round(frame_h*20/543)
            line_space=int(text_size/4)
            xl,yl,xr,yr = pil_drawing.textbbox((0,0), log_str_up, spacing=line_space,font=ImageFont.truetype("consolai.ttf", text_size))
            loc_h=round(frame_h*0.025)
            loc_w=frame_w-loc_h-xr+xl
            
            pil_drawing.text((loc_w,loc_h), log_str_up, align='left',fill=(255, 255, 255),font=ImageFont.truetype("consolai.ttf", text_size), spacing=line_space)
            pil_drawing.text((loc_w,loc_h+yr-yl+line_space), log_str_down, align='left',fill=T_color,font=ImageFont.truetype("consolai.ttf", text_size))
            
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        def save_img():
            save_path=self.img_save_path.get()
            extent_name=os.path.splitext(save_path)[-1]
            if extent_name in img_type:
                required_size=self.save_size.get()
                if  self.res_cv_image.shape[:2] != required_size[::-1]:
                    save_img=cv2.resize(self.res_cv_image,required_size)
                else:
                    save_img=self.res_cv_image
                
                if self.put_data.get():
                    save_img=save_detect_data(save_img)
                
                cv2.imwrite(save_path,save_img)
                img_save_bar.configure(value=1)
                bt_opendir.configure(state='normal')
            else:
                delete_img_path()
                mb.showerror(title='保存错误',message=f'请重新指定保存路径\n确保图片扩展名正确\n\n目前仅支持以下图片类型\n{img_type}')                

        def save_video():
            global cancel_export
            cancel_export=0
            
            def stop():
                global cancel_export
                cancel_export=1

            def export(format,save_size,required_size,if_resize):
                global cancel_export
                
                video_save_bar.configure(maximum=self.region_info[1]-self.region_info[0]+1)
                vid_writer = cv2.VideoWriter(save_path, format,self.save_fps.get(), save_size)
                for now_value,img_idx in enumerate(range(self.region_info[0]-1,self.region_info[1])):
                    if cancel_export:
                        vid_writer.release()
                        bt_opendir.configure(state='normal')
                        bt_save_video.configure(text='导 出 视 频',command=save_video)
                        break
                    
                    self.bar_val.set(img_idx+1)
                    _=self.get_new_frame(img_idx+1)

                    if if_resize:
                        save_img=cv2.resize(self.res_cv_image,required_size)
                    else:
                        save_img=self.res_cv_image

                    if self.put_data.get():
                        save_img=save_detect_data(save_img)

                    vid_writer.write(save_img)

                    video_save_bar.configure(value=now_value+1)
                    video_save_bar.update()
                    sleep(0.01)
                else:
                    cancel_export=1
                    vid_writer.release()
                    bt_opendir.configure(state='normal')
                    bt_save_video.configure(text='导 出 视 频',command=save_video)

            save_path=self.video_save_path.get()
            extent_name=os.path.splitext(save_path)[-1]
            if extent_name not in video_type:
                delete_video_path()
                mb.showerror(title='保存错误',message=f'请重新指定保存路径\n确保视频扩展名正确\n\n目前仅支持以下视频类型\n{video_type}') 
            else:
                required_size=self.save_size.get()
                now_shape=tuple(self.res_cv_image.shape[:2][::-1])
                if required_size!=now_shape:
                    save_size=required_size
                    if_resize=1
                else:
                    save_size=now_shape
                    if_resize=0

                if extent_name=='.mp4':
                    format=cv2.VideoWriter_fourcc(*'mp4v')
                elif extent_name=='.avi':
                    format=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                else:
                    format=cv2.VideoWriter_fourcc(*'WMV3')
                
                bt_save_video.configure(text='暂 停 导 出',command=stop)
                
                Thread(target=export,args=(format,save_size,required_size,if_resize)).start()

        def set_size():
            new_w,new_h=map(int,self.save_size.get().split(' '))
            self.save_size.set([new_w,new_h])

        def popup_else_size():
            img_save_bar.configure(value=0)
            now_frame_h,now_frame_w=self.res_cv_image.shape[:2]
            now_rate=now_frame_w/now_frame_h
            rate_dis=abs(now_rate-np.array([16/9,1.25,4/3,1.5,256/135]))
            close_rate_idx=np.argmin(rate_dis)
            if rate_dis[close_rate_idx]>0.1:    # the situation that match none the given size
                close_rate_idx=10

            size_win=tk.Toplevel(self.gui)
            size_win.title('选择帧大小')
            size_win.resizable(False,False)
            size_win.attributes('-topmost', 'true')
            
            tk.Label(size_win, text='16 : 9').grid(row=0,column=0,sticky='ew')
            size_19201080=tk.Radiobutton(size_win, text='1920×1080',fg='red', variable=self.save_size,value=[1920,1080],command=set_size)
            size_19201080.grid(row=0,column=1,sticky='w')
            size_1280720=tk.Radiobutton(size_win, text='1280×720', fg='red',variable=self.save_size, value=[1280,720],command=set_size) 
            size_1280720.grid(row=0,column=2,columnspan=2,sticky='w')
            size_38402160=tk.Radiobutton(size_win, text='UHD: 3840×2160', fg='red',variable=self.save_size, value=[3840,2160],command=set_size)
            size_38402160.grid(row=4,column=3,sticky='w')

            if close_rate_idx==0:
                size_19201080.configure(fg='black')
                size_1280720.configure(fg='black')
                size_38402160.configure(fg='black')

            tk.Label(size_win, text='5 : 4').grid(row=1,column=0,sticky='ew')
            size_720576=tk.Radiobutton(size_win, text='720×576', fg='red',variable=self.save_size, value=[720,576],command=set_size)
            size_720576.grid(row=1,column=1,columnspan=3,sticky='w')

            if close_rate_idx==1:
                size_720576.configure(fg='black')
            
            tk.Label(size_win, text='4 : 3').grid(row=2,column=0,sticky='ew')
            size_960720=tk.Radiobutton(size_win, text='960×720', fg='red',variable=self.save_size, value=[960,720],command=set_size)
            size_960720.grid(row=2,column=1,sticky='w')
            size_640480=tk.Radiobutton(size_win, text='640×480', fg='red',variable=self.save_size, value=[640,480],command=set_size)
            size_640480.grid(row=2,column=2,sticky='w')
            size_14401080=tk.Radiobutton(size_win, text='1440×1080', fg='red',variable=self.save_size, value=[1440,1080],command=set_size)
            size_14401080.grid(row=2,column=3,sticky='w')

            if close_rate_idx==2:
                size_960720.configure(fg='black')
                size_640480.configure(fg='black')
                size_14401080.configure(fg='black')
            
            tk.Label(size_win, text='3 : 2').grid(row=3,column=0,sticky='ew')
            size_720480=tk.Radiobutton(size_win, text='720×480', fg='red',variable=self.save_size, value=[720,480],command=set_size)
            size_720480.grid(row=3,column=1,columnspan=3,sticky='w')

            if close_rate_idx==3:
                size_720480.configure(fg='black')

            tk.Label(size_win, text='超清').grid(row=4,column=0,sticky='ew')
            size_20481080=tk.Radiobutton(size_win, text='2k: 2048×1080', fg='red',variable=self.save_size, value=[2048,1080],command=set_size) 
            size_20481080.grid(row=4,column=1,sticky='w')
            size_40962160=tk.Radiobutton(size_win, text='4k: 4096×2160', fg='red',variable=self.save_size, value=[4096,2160],command=set_size) 
            size_40962160.grid(row=4,column=2,sticky='w')
            
            if close_rate_idx==4:
                size_20481080.configure(fg='black')
                size_40962160.configure(fg='black')

            tk.Label(size_win,text="提示: 红色选项表示该尺寸会破坏现有图像比例，请谨慎选择",\
                 fg='#808080',anchor='w').grid(padx=5,row=5,column=0,columnspan=4,sticky='w')

            grey_buttom=tk.Frame(size_win,bg='#D0D0D0',height=30)
            grey_buttom.pack_propagate(0)
            grey_buttom.grid(row=6,column=0,columnspan=4,sticky='news')
            tk.Button(grey_buttom,text='确\t定',command=size_win.destroy,bg='#D0D0D0', bd=1,relief='flat').pack(fill='both',expand=True)

        def cancel():
            global cancel_export
            try:       # to close the window when there is no export action
                if cancel_export==0:
                    mb.showerror(title='窗口关闭错误',message='视频正在导出中')
                    return
            except:
                pass

            self.bt_play.configure(state='normal')
            self.bt_close.configure(state='normal')
            self.bt_detect.configure(state='normal')

            self.img_save_path.set('')
            self.video_save_path.set('')

            self.save_size.set([now_frame_w,now_frame_h])

            save_win.destroy()

        save_win=tk.Toplevel(self.gui)
        save_win.title('导出结果')
        save_win.resizable(False,False)
        save_win.attributes('-topmost', 'true')
        save_win.protocol("WM_DELETE_WINDOW",cancel)

        now_frame_w,now_frame_h=self.save_size.get()
        self.bt_play.configure(state='disabled')
        self.bt_close.configure(state='disabled')
        self.bt_detect.configure(state='disabled')

        main_frame=tk.Frame(save_win)
        main_frame.pack_propagate(0)
        main_frame.pack(padx=5,pady=5,fill='both',expand=True)

        path_frame=tk.Frame(main_frame)
        path_frame.grid(row=0,column=0,sticky='news')
        path_frame.columnconfigure(0,weight=2)
        path_frame.columnconfigure(1,weight=27)
        path_frame.columnconfigure(2,weight=1)
        tk.Label(path_frame, text='图片保存 至').grid(row=0,column=0,sticky='ew')
        tk.Entry(path_frame,textvariable=self.img_save_path,state='readonly',readonlybackground='white',bd=2,relief='groove').grid(row=0,column=1,sticky='ew',padx=5,ipadx=0)
        tk.Button(path_frame,text='浏览',command=choose_img_dir, cursor='hand2',bd=2,relief='groove').grid(row=0,column=2,sticky='news')

        tk.Label(path_frame, text='视频保存 至').grid(row=1,column=0,sticky='ew')
        tk.Entry(path_frame,textvariable=self.video_save_path,state='readonly',readonlybackground='white',bd=2,relief='groove').grid(row=1,column=1,sticky='ew',padx=5,ipadx=0)
        tk.Button(path_frame,text='浏览',command=choose_video_dir, cursor='hand2',bd=2,relief='groove').grid(row=1,column=2,sticky='news')
        
        tk.Frame(main_frame, height=2, bg='light grey').grid(row=1,column=0,sticky='ew',pady=10)

        put_data_frame=tk.Frame(main_frame)
        put_data_frame.columnconfigure(0,weight=1)
        put_data_frame.columnconfigure(1,weight=1)
        put_data_frame.columnconfigure(2,weight=1)
        put_data_frame.grid(row=2,column=0,sticky='news')
        tk.Label(put_data_frame, text='记录检测结果:').grid(row=0,column=0,sticky='w')
        tk.Radiobutton(put_data_frame, text='是', variable=self.put_data, value=1).grid(row=0,column=1,sticky='w')
        tk.Radiobutton(put_data_frame, text='否', variable=self.put_data, value=0).grid(row=0,column=2,sticky='w')

        fps_frame=tk.Frame(main_frame)
        fps_frame.columnconfigure(0,weight=1)
        fps_frame.columnconfigure(1,weight=1)
        fps_frame.columnconfigure(2,weight=28)
        tk.Label(fps_frame, text='帧率:').grid(row=1,column=0,sticky='w')
        tk.Label(fps_frame,textvariable=self.save_fps).grid(row=1,column=1,sticky='w')
        tk.Scale(fps_frame,from_=20,to=30,variable=self.save_fps, showvalue=0,orient=tk.HORIZONTAL).grid(row=1,column=2,sticky='ew')

        size_frame=tk.Frame(main_frame)
        size_frame.grid(row=4,column=0,sticky='news')
        size_frame.columnconfigure(0,weight=1)
        size_frame.columnconfigure(1,weight=1)
        size_frame.columnconfigure(2,weight=28)
        tk.Label(size_frame, text='帧大小(宽×高): ').grid(row=0,column=0,sticky='w')
        tk.Radiobutton(size_frame, text=f'当前大小({now_frame_w}×{now_frame_h})', variable=self.save_size, value=[now_frame_w,now_frame_h],command=set_size).grid(row=0,column=1,sticky='ew')
        tk.Button(size_frame, text='( 选 择 其 他 尺 寸 )', command=popup_else_size, cursor='hand2',bd=2,relief='groove').grid(row=0,column=2,sticky='ew')

        tk.Frame(main_frame, height=2, bg='light grey').grid(row=5,column=0,sticky='ew',pady=10)

        save_frame=tk.Frame(main_frame)
        save_frame.grid(row=6,column=0,sticky='news',padx=5)
        save_frame.columnconfigure(0,weight=1)
        save_frame.columnconfigure(1,weight=1)
        save_frame.columnconfigure(2,weight=4)
        save_frame.columnconfigure(3,weight=4)
        bt_save_img=tk.Button(save_frame,text='保 存 截 图',command=save_img,bd=2,relief='groove',state='disabled')
        bt_save_img.grid(row=0,column=0,columnspan=2,sticky='ew',pady=5)
        bt_save_video=tk.Button(save_frame,text='导 出 视 频',command=save_video,bd=2,relief='groove',state='disabled')
        bt_save_video.grid(row=0,column=2,columnspan=2,sticky='ew',pady=5)
        if self.img_save_path.get():
            bt_save_img.configure(state='normal')
        if self.video_save_path.get():
            bt_save_video.configure(state='normal')
        
        img_info_frame=tk.Frame(save_frame)
        img_info_frame.grid(row=1,column=0,columnspan=2,sticky='ew')
        tk.Label(img_info_frame,text='帧位置:',fg='red').pack(side='left')
        tk.Label(img_info_frame,textvariable=self.bar_val,foreground='red',anchor='nw').pack(side='left')
        img_save_bar=Progressbar(img_info_frame,value=0, maximum=1,orient=tk.HORIZONTAL)
        img_save_bar.pack(side='left',fill='x',expand=True)
        
        video_info_frame=tk.Frame(save_frame)
        video_info_frame.grid(row=1,column=2,columnspan=2,sticky='ew')
        tk.Label(video_info_frame,text='帧区间:',fg='red').pack(side='left')
        tk.Label(video_info_frame,textvariable=self.region_tips_str,fg='red').pack(side='left')
        video_save_bar=Progressbar(video_info_frame,value=0, maximum=self.region_info[1]-self.region_info[0],orient=tk.HORIZONTAL)
        video_save_bar.pack(side='left',fill='x',expand=True)
        
        tk.Frame(main_frame, height=2, bg='light grey').grid(row=7,column=0,sticky='ew',pady=10)

        tk.Label(main_frame,text="提示:",\
                 fg='#808080',anchor='w').grid(padx=5,row=8,column=0,sticky='w')
        tk.Label(main_frame,text='“当前大小”仅代表打开此窗口时的图像大小，无法表示实时图像大小',\
                 fg='#808080',anchor='w').grid(padx=5,row=9,column=0,sticky='w')
        tk.Label(main_frame,text='请检查所导出的帧区间，若需调整，请点击左下角 “设置区间” 按钮',\
                 fg='#808080',anchor='w').grid(padx=5,row=10,column=0,sticky='w')

        buttom_margin=tk.Frame(save_win,bg='#D0D0D0',height=40)
        buttom_margin.pack_propagate(0)
        buttom_margin.pack(fill='both',expand=True)
        tk.Button(buttom_margin,text='设置区间',command=self.set_region,bg='#D0D0D0', bd=1,relief='flat').pack(padx=10,side='left')
        tk.Button(buttom_margin,text='取消',command=cancel,bg='#D0D0D0', bd=1,relief='flat').pack(padx=10,side='right')
        bt_opendir=tk.Button(buttom_margin,text='打开文件位置',command=open_dir,state='disabled',bg='#D0D0D0',bd=1,relief='flat')
        bt_opendir.pack(padx=10,side='right')

        save_win.mainloop()

    def general_setting(self):
        set_win=tk.Toplevel(self.gui)
        set_win.title('软件设置')
        set_win.resizable(False,False)
        set_win.attributes('-topmost', 'true')
        set_win.attributes('-toolwindow', 'true') # mini window without icon
        
        def choose_model():
            choose_path=filedialog.askopenfilename(
                title='选择YoLoV5模型',
                defaultextension='.pt',
                filetypes=[('模型文件',('.pt','.pth'))]
            )
            if not choose_path:
                return 
            else:
                self.weight_path.set(choose_path)
                path_state.configure(text='✔')

        def choose_npy_file():
            choose_path=filedialog.askopenfilename(
                title='选择相机内参文件',
                defaultextension='.npy',
                filetypes=[('内参文件',('.npy'))]
            )
            if not choose_path:
                mb.showinfo(title='提醒',message='若未进行相机标定\n请前往  菜单栏--相机标定  上传标定图片，并保存内参文件')
                return 
            else:
                self.npy_path.set(choose_path)
                self.intri_mat=np.load(self.npy_path.get())
                npy_path_state.configure(text='✔')

        def disable_compress():
            self.close_media()
            compress_frame.grid_forget()

        def activate_compress():
            self.close_media()
            popup_compress_grade()

        def set_compress_grade(_):
            self.compress_params[1]=self.compress_grade.get()
            #print(self.compress_params)

        def popup_compress_grade():
            compress_frame.grid(row=2,column=0,sticky='news',pady=5)
            compress_frame.columnconfigure(0,weight=1)
            compress_frame.columnconfigure(1,weight=1)
            compress_frame.columnconfigure(2,weight=18)
            
            tk.Label(compress_frame,text='帧压缩等级:').grid(pady=5,row=0,column=0,sticky='news')
            grade_show.grid(pady=5,row=0,column=1,sticky='news')
            compress_bar.grid(pady=5,row=0,column=2,sticky='ew')

            tk.Label(compress_frame,text='  提示:  压缩等级越大，帧数据占用内存更小，但帧质量更低，加载更慢',\
                 fg='#808080',anchor='w').grid(row=2,column=0,columnspan=3,sticky='w')

        def disable_gpu():
            self.close_media()
            device_frame.grid_forget()

        def assert_gpu():
            cuda_choice=list(map(lambda x:x.get(),self.cuda_var))
            if 1 not in cuda_choice:
                mb.showwarning(title='操作取消',message='必须选择至少一张GPU\n若不想使用GPU计算，请选择 CPU 项')
                self.cuda_var[0].set(1)
                cuda_choice[0]=1

        def popup_gpu_choose():
            column=2
            device_frame.columnconfigure(0,weight=1)
            device_frame.columnconfigure(1,weight=1)
            device_frame.grid(row=1,column=0,columnspan=3,sticky='ew')

            for gpu_idx, gpu_var in enumerate(self.cuda_var):
                tk.Checkbutton(device_frame, text=torch.cuda.get_device_name(gpu_idx), variable=gpu_var, onvalue=gpu_idx+1, offvalue=0,command=assert_gpu).grid(row=gpu_idx//column, column=gpu_idx%column,columnspan=2, sticky='ew')

        main_frame=tk.Frame(set_win)
        main_frame.pack_propagate(0)
        main_frame.pack(padx=5,pady=5,fill='both',expand=True)

        load_frame=tk.Frame(main_frame)
        load_frame.columnconfigure(0,weight=1)
        load_frame.columnconfigure(1,weight=1)
        load_frame.columnconfigure(2,weight=1)
        load_frame.grid(row=0,column=0,sticky='w')
        tk.Label(load_frame,text='媒体加载方式:').grid(row=0,column=0,sticky='w')
        tk.Radiobutton(load_frame, text='实时', variable=self.frame_source, value='realtime',command=disable_compress).grid(row=0,column=1,sticky='w',padx=5)
        tk.Radiobutton(load_frame, text='内存', variable=self.frame_source, value='memory',command=activate_compress).grid(row=0,column=2,sticky='w',padx=5)
        tk.Label(load_frame,text='摄像画面翻转:').grid(row=1,column=0,sticky='w')
        tk.Radiobutton(load_frame, text='是', variable=self.flip_cam, value=1).grid(row=1,column=1,sticky='w',padx=5)
        tk.Radiobutton(load_frame, text='否', variable=self.flip_cam, value=0).grid(row=1,column=2,sticky='w',padx=5)

        compress_frame=tk.Frame(main_frame,bd=2,relief='groove')
        grade_show=tk.Label(compress_frame,textvariable=self.compress_grade)
        compress_bar=tk.Scale(compress_frame,from_=0,to=9,variable=self.compress_grade,command=set_compress_grade,showvalue=0,orient=tk.HORIZONTAL)
        if self.frame_source.get()=='memory':
            popup_compress_grade()

        tk.Frame(main_frame, height=2, bg='light grey').grid(row=3,column=0,sticky='ew',pady=5)

        detect_params_frame=tk.Frame(main_frame)
        detect_params_frame.columnconfigure(0,weight=1)
        detect_params_frame.columnconfigure(1,weight=1)
        detect_params_frame.columnconfigure(2,weight=58)
        detect_params_frame.grid(row=4,column=0,sticky='news')
        tk.Label(detect_params_frame,text='运 行 平 台:').grid(row=0,column=0,sticky='w')
        device_frame=tk.Frame(detect_params_frame,bd=2,relief='groove')
        tk.Radiobutton(detect_params_frame, text='CPU', variable=self.if_cpu, value=1,command=disable_gpu).grid(row=0,column=1,sticky='w')
        if self.cuda_device_num:
            tk.Radiobutton(detect_params_frame, text='GPU', variable=self.if_cpu, value=0, command=popup_gpu_choose).grid(row=0,column=2,sticky='w')
        if not self.if_cpu.get():
            popup_gpu_choose()
        tk.Label(detect_params_frame,text='预处理尺寸:').grid(row=2, column=0, sticky='w')
        tk.Label(detect_params_frame,textvariable=self.img_size).grid(row=2,column=1,sticky='news')
        tk.Scale(detect_params_frame,from_=320,to=1088, variable=self.img_size,resolution=32,showvalue=0,orient=tk.HORIZONTAL).grid(row=2,column=2,sticky='ew')
        tk.Label(detect_params_frame,text='置信度阈值:').grid(row=3, column=0, sticky='w')
        tk.Label(detect_params_frame,textvariable=self.conf).grid(row=3,column=1,sticky='news')
        tk.Scale(detect_params_frame,from_=0,to=1, variable=self.conf,resolution=0.01 ,showvalue=0,orient=tk.HORIZONTAL).grid(row=3,column=2,sticky='ew')
        tk.Label(detect_params_frame,text='交并比阈值:').grid(row=4, column=0, sticky='w')
        tk.Label(detect_params_frame,textvariable=self.iou).grid(row=4,column=1,sticky='news')
        tk.Scale(detect_params_frame,from_=0,to=1, variable=self.iou,resolution=0.01,showvalue=0,orient=tk.HORIZONTAL).grid(row=4,column=2,sticky='ew')
        tk.Label(detect_params_frame,text='低速值(km/h):').grid(row=5, column=0, sticky='w')
        tk.Label(detect_params_frame,textvariable=self.vt).grid(row=5,column=1,sticky='news')
        tk.Scale(detect_params_frame,from_=1,to=70, variable=self.vt,showvalue=0,orient=tk.HORIZONTAL).grid(row=5,column=2,sticky='ew')

        tk.Frame(main_frame, height=2, bg='light grey').grid(row=5,column=0,sticky='ew',pady=5)

        path_frame=tk.Frame(main_frame)
        path_frame.grid(row=6,column=0,sticky='nesw')
        tk.Label(path_frame,text='YoLoV5模型路径:').grid(row=0,column=0,sticky='news')
        tk.Entry(path_frame,textvariable=self.weight_path,state='readonly',readonlybackground='white',width=40,bd=2,relief='groove').grid(row=0,column=1,sticky='w',padx=5,pady=5)
        path_state=tk.Label(path_frame,text='❌')
        if os.path.splitext(self.weight_path.get())[1] in ['.pt','.pth']:
            path_state.configure(text='✔')
        path_state.grid(row=0,column=2,sticky='news')
        tk.Button(path_frame,text='浏览',command=choose_model,bd=2,relief='groove').grid(row=0,column=3,sticky='news')

        tk.Label(path_frame,text='相机内参文件路径:').grid(row=1,column=0,sticky='news')
        tk.Entry(path_frame,textvariable=self.npy_path,state='readonly',readonlybackground='white',width=40,bd=2,relief='groove').grid(row=1,column=1,sticky='w',padx=5)
        npy_path_state=tk.Label(path_frame,text='❌')
        if os.path.splitext(self.npy_path.get())[1]=='.npy':
            npy_path_state.configure(text='✔')
        npy_path_state.grid(row=1,column=2,sticky='news')
        tk.Button(path_frame,text='浏览',command=choose_npy_file,bd=2,relief='groove').grid(row=1,column=3,sticky='news')
        
        buttom_margin=tk.Frame(set_win,bg='#D0D0D0',height=40)
        buttom_margin.pack_propagate(0)
        buttom_margin.pack(fill='both',expand=True)
        tk.Button(buttom_margin,text='确\t\t定',command=set_win.destroy,bg='#D0D0D0', bd=1,relief='flat',activebackground='red').pack(fill='both',expand=True)

    def calibration_setting(self):
        calib_win=tk.Toplevel(self.gui)
        calib_win.title('相机标定设置')
        calib_win.resizable(False,False)
        calib_win.attributes('-topmost', 'true')

        def choose_dir():
            choose_path=filedialog.askdirectory(
                title='选择标定图片所在文件夹',
            )
            if not choose_path:
                return 
            else:
                img_type=('.jpg','.png','.jpeg','.bmp','.webp')
                img_num=sum(list(map(lambda x:os.path.splitext(x)[-1] in img_type, os.listdir(choose_path))))
                if img_num>20:
                    self.calib_img_path.set(choose_path)
                    bt_calib.configure(state='normal')
                else:
                    mb.showwarning(title='警告',message=f'所选文件夹内含图片: {img_num} 张\n图片数量过少，请保证标定图片数大于 20 张！')

        def calibration():
            lb_valid_num.grid_forget()
            lb_error.grid_forget()
            lb_more.grid_forget()
            process_bar.configure(mode='indeterminate')

            tk.Label(calib_frame,text='处理中:').grid(row=4, column=0, sticky='nesw',padx=5,pady=5)
            process_bar.grid(row=4, column=1,columnspan=2,sticky='news',padx=5,pady=5)
            process_bar.start(30)
            
            def start_calibrate():
                resize_path=correct_img_size(self.calib_img_path.get())
                valid_num,*calib_params=camera_calibration(resize_path, self.board_column.get()-1, self.board_row.get()-1, draw_corners=True)
                if valid_num==0:
                    process_bar.stop()
                    mb.showerror(title='错误',message='未能在图片中识别到所设行列数的棋盘格\n\n你可以：\n1. 检查行列数设置；\n2. 确保标定板无遮挡、反光现象，重新标定')
                    return 

                self.npy_path.set(calib_params[0])
                img_path_list, world_points, img_points,intri_mat, dist, rvecs, tvecs=calib_params[1]
                self.intri_mat=intri_mat
                total_error, mean_error=get_reproject_error(world_points, img_points, intri_mat, dist, rvecs, tvecs, img_path_list=img_path_list)
                process_bar.stop()
                process_bar.configure(value=100,mode='determinate')
            
                lb_valid_num.configure(text=f'有效识别图片数: {valid_num}')
                lb_error.configure(text='重投影误差    总误差 -> {:.3f}\t均值误差 -> {:.3f}'.format(total_error,mean_error))
                lb_more.configure(text='内参文件 及 详细标定结果，可点击 “打开文件位置” 查看')

                lb_valid_num.grid(row=5,column=0,columnspan=3,sticky='w')
                lb_error.grid(row=6, column=0, columnspan=3, sticky='w')
                lb_more.grid(row=7,column=0,columnspan=3,sticky='w')
                bt_opendir.configure(state='normal')
                
            Thread(target=start_calibrate,args=()).start()
            
        def open_dir():
            dir=os.path.split(self.npy_path.get())[0]
            os.system(f'start {dir}')
        
        main_frame=tk.Frame(calib_win)
        main_frame.pack_propagate(0)
        main_frame.pack(padx=5,pady=5,fill='both',expand=True)

        path_frame=tk.Frame(main_frame)
        path_frame.grid(row=0,column=0,columnspan=2,sticky='nesw')
        tk.Label(path_frame,text='标定图片所在路径:').grid(row=0,column=0,sticky='news')
        tk.Entry(path_frame,textvariable=self.calib_img_path,state='readonly',readonlybackground='white',width=40,bd=2,relief='groove').grid(row=0,column=1,sticky='ew',padx=5)
        tk.Button(path_frame,text='浏览',command=choose_dir,bd=2,relief='groove').grid(row=0,column=3,sticky='news')

        calib_frame=tk.Frame(main_frame)
        calib_frame.columnconfigure(0,weight=1)
        calib_frame.columnconfigure(1,weight=1)
        calib_frame.columnconfigure(2,weight=38)
        calib_frame.grid(row=1,column=0,columnspan=2,sticky='nesw')
        tk.Label(calib_frame,text='棋盘行数:').grid(row=0, column=0, sticky='w')
        tk.Label(calib_frame,textvariable=self.board_row).grid(row=0,column=1,sticky='news')
        tk.Scale(calib_frame,from_=0,to=20, variable=self.board_row,showvalue=0,orient=tk.HORIZONTAL).grid(row=0,column=2,sticky='ew')
        tk.Label(calib_frame,text='棋盘列数:').grid(row=1, column=0, sticky='w')
        tk.Label(calib_frame,textvariable=self.board_column).grid(row=1,column=1,sticky='news')
        tk.Scale(calib_frame,from_=0,to=20, variable=self.board_column,showvalue=0,orient=tk.HORIZONTAL).grid(row=1,column=2,sticky='ew')
        tk.Label(calib_frame,text='请仔细确定行数和列数，其将直接影响标定结果！！！',fg='red').grid(row=2, column=0, columnspan=3, sticky='w')
        bt_calib=tk.Button(calib_frame,text='开 始 标 定',command=calibration,state='disable',bd=2,relief='groove')
        bt_calib.grid(row=3,column=0,columnspan=3,sticky='news')
        if self.calib_img_path.get():
            bt_calib.configure(state='normal')
        process_bar=Progressbar(calib_frame, mode='indeterminate', maximum=100,orient=tk.HORIZONTAL)
        lb_valid_num=tk.Label(calib_frame)
        lb_error=tk.Label(calib_frame)
        lb_more=tk.Label(calib_frame)

        tk.Frame(main_frame, height=2, bg='light grey').grid(row=2,column=0,columnspan=2,sticky='ew',pady=5)

        tk.Label(main_frame,text="提示:",\
                 fg='#808080',anchor='w').grid(padx=5,row=3,column=0,columnspan=2,sticky='w')
        tk.Label(main_frame,text='右图棋盘格图片\n\n行数可设置为3\n\n列数可设置为5',\
                 fg='#808080',font=('微软雅黑',10)).grid(padx=5,row=4,column=0,sticky='ew')
        example_canvas=tk.Canvas(main_frame,width=250,height=150,bd=1,relief='flat')
        example_canvas.grid(padx=5,row=4,column=1,sticky='w')
        example_canvas.create_rectangle( 0, 0,50,50,fill='black')
        example_canvas.create_rectangle(50,0,100,50,fill='white')
        example_canvas.create_rectangle(100,0,150,50,fill='black')
        example_canvas.create_rectangle(150,0,200,50,fill='white')
        example_canvas.create_rectangle(200,0,250,50,fill='black')

        example_canvas.create_rectangle( 0,50,50,100,fill='white')
        example_canvas.create_rectangle(50,50,100,100,fill='black')
        example_canvas.create_rectangle(100,50,150,100,fill='white')
        example_canvas.create_rectangle(150,50,200,100,fill='black')
        example_canvas.create_rectangle(200,50,250,100,fill='white')

        example_canvas.create_rectangle( 0,100,50,150,fill='black')
        example_canvas.create_rectangle(50,100,100,150,fill='white')
        example_canvas.create_rectangle(100,100,150,150,fill='black')
        example_canvas.create_rectangle(150,100,200,150,fill='white')
        example_canvas.create_rectangle(200,100,250,150,fill='black')

        buttom_margin=tk.Frame(calib_win,bg='#D0D0D0',height=40)
        buttom_margin.pack_propagate(0)
        buttom_margin.pack(fill='both',expand=True)
        bt_opendir=tk.Button(buttom_margin,text='打开文件位置',command=open_dir,state='disable',bg='#D0D0D0',bd=1,relief='flat')
        tk.Button(buttom_margin,text='取消',command=calib_win.destroy,bg='#D0D0D0',bd=1,relief='flat').pack(side='right',padx=5)
        bt_opendir.pack(side='right',padx=5)

if __name__=='__main__':
    gui=TPZ_GUI()