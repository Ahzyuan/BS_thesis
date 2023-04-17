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
      'gui_title':'基于YoLoV5的车辆自动礼让行人算法 (by 黄子源)',
      'gui_w':'1300',
      'gui_h':'600'
    }
}