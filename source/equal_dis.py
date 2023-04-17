import cv2

img_path=r'E:\AIL\project\Undergraduate_Thesis\source\dis_test\75.jpg'
img=cv2.imread(img_path)
w=img.shape[1]

dis=list(range(5,76,5))
pix_loc=map(lambda x:1862.77/(x+1.8)+538.3, dis)

for loc in pix_loc:
    loc=round(loc)
    cv2.line(img,(0,loc),(w,loc),color=(0,0,255),thickness=2)

cv2.imwrite(r'E:\AIL\project\Undergraduate_Thesis\source\equal_dis.jpg',img)