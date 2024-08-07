import numpy as np

config={
    'hsv_color_range':
    {   
      'Red1':(np.array([0,43,46]), np.array([10,255,255])),    # hsv value:min,max
      'Red2':(np.array([156,43,46]), np.array([180,255,255])),
      'Green':(np.array([35,43,46]), np.array([89,255,255])),
      'Yellow':(np.array([11,43,46]), np.array([34,255,255]))
    },

    'gui_info':{
      'gui_title':'基于多目标检测与跟踪的车辆自动礼让行人算法研究与应用 (by 黄子源 姚志强)',
      'gui_w':'1300',
      'gui_h':'600',
      'font_color':{   #the color of the font which shows the obj's distance
            'R':'#C00000',
            'G':'#00A44A',
            'B':'#000000',
            'Y':'#EEC600'
        }
    }
}