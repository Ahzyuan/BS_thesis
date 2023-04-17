import cv2
import numpy as np

def hsv2rgb(h,s,v):
    color=np.uint8([[[h,s,v]]])
    b,g,r=cv2.cvtColor(color,cv2.COLOR_HSV2BGR)[0][0]
    print('h {},s {},v {} -> r {},g {},b {}'.format(h,s,v,r,g,b))

def rgb2hsv(r,g,b):
    color=np.uint8([[[b,g,r]]])
    h,s,v=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)[0][0]
    print('r {},g {},b {} -> h {},s {},v {}'.format(r,g,b,h,s,v))