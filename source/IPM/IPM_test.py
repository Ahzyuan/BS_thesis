import cv2
import numpy as np
 
def multi_transform(img, pts1):
 
    ROI_HEIGHT = 30000
    ROI_WIDTH = 3750
 
    # 设定逆透视图的宽度
    IPM_WIDTH = 600
    N = 5
 
    # 保证逆透视图的宽度大概为N个车头宽
    sacale=(IPM_WIDTH/N)/ROI_WIDTH
    IPM_HEIGHT=250 #ROI_HEIGHT*sacale
 
    pts2 = np.float32([[IPM_WIDTH/2-IPM_WIDTH/20, 0],
                       [IPM_WIDTH/2+IPM_WIDTH/20, 0],
                       [IPM_WIDTH/2-IPM_WIDTH/20, IPM_HEIGHT],
                       [IPM_WIDTH/2+IPM_WIDTH/20, IPM_HEIGHT]])
 
    print(IPM_HEIGHT,IPM_WIDTH)
 
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(img, matrix, (int(IPM_WIDTH),int(IPM_HEIGHT)))

    for i in range(0, 4):
        cv2.circle(img, (int(pts1[i][0]), int(pts1[i][1])), 6, (0, 0, 255), cv2.FILLED)
    
    for i in range(0,4):
        cv2.circle(output, (int(pts2[i][0]), int(pts2[i][1])),6, (0, 0, 255), cv2.FILLED)
 
    # p1 = (0, 250)
    # p2 = (img.shape[1], img.shape[0]-100)
    # point_color = (255, 0, 0)
    # cv2.rectangle(img, p1, p2, point_color, 2)
 
    cv2.imshow("src image", img)
    cv2.imshow("output image", output)
    cv2.waitKey(0)
 
if __name__ == '__main__':
    # 图像1
    #img = cv2.imread("1.jpg")
    #pts1 = np.float32([[321, 250],       # p1 left_top (column, row)
    #                   [408, 250],       # p2 right_top
    #                   [241, 393],       # p3 left_buttom
    #                   [477, 393]])      # p4 right_buttom
 
    # 图像2
    img = cv2.imread("2.png")
    pts1 = np.float32([[129, 0],       # p1
                       [208, 0],       # p2
                       [0, 221],       # p3
                       [313, 221]])      # p4
 
    multi_transform(img, pts1)