import cv2
import numpy as np
import time


def callback(object):
    pass


cv2.namedWindow('mask_yellow', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('mask_green', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('mask_red', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('red1_H', 'mask_red',0,10,callback)#red1
cv2.createTrackbar('red2_H', 'mask_red',156,180,callback)#red1
cv2.createTrackbar('red_S','mask_red',43,255,callback)
cv2.createTrackbar('red_V','mask_red',200,255,callback)

cv2.createTrackbar('green_H', 'mask_green',35,77,callback)#red1
cv2.createTrackbar('green_S', 'mask_green',43,255,callback)
cv2.createTrackbar('green_V', 'mask_green',200,255,callback)

cv2.createTrackbar('yellow_H', 'mask_yellow',11,43,callback)#red1
cv2.createTrackbar('yellow_S', 'mask_yellow',43,255,callback)
cv2.createTrackbar('yellow_V', 'mask_yellow',200,255,callback)

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

green_low = np.array([40, 103, 164])
green_up = np.array([77, 255, 255])

red0_low = np.array([5, 166, 224])
red1_low = np.array([160, 166,224])

red0_up = np.array([10, 255, 255])
red1_up = np.array([180, 255, 255])

lower_yellow = np.array([15, 130, 190])  # 黄色低阈值
upper_yellow = np.array([34, 255, 255])  # 黄色高阈值

def color_rec(dst):
    color=0
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, green_low, green_up)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_mask1 = cv2.inRange(hsv, red0_low, red0_up)
    red_mask2 = cv2.inRange(hsv, red1_low, red1_up)
    mask_red = cv2.bitwise_or(red_mask1, red_mask2)

    contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours1, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy3 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours):
        (x, y, w, h) = cv2.boundingRect(contours[0])

    else:
        w=0
        h=0

    if len(contours1):
        (x1, y1, w1, h1) = cv2.boundingRect(contours1[0])
    else:
        w1=0
        h1=0

    if len(contours2):
        (x2, y2, w2, h2) = cv2.boundingRect(contours2[0])
    else:
        w2=0
        h2=0

    s1=w*h
    s2=w1*h1
    s3=w2*h2
    if s1>=30:
        print("green")
        color=1    #green
    elif s2>=30:
        print('red')
        color=2    #red
    elif s3>=30:
        print('yellow')
        color=3    #yelow
    cv2.imshow("mask_red", mask_red)
    cv2.imshow("mask_yellow", mask_yellow)
    cv2.imshow("mask_green", mask_green)
    return color

while True:
    ret, frame = capture.read()
    t1=time.time()
    H1 = cv2.getTrackbarPos('red1_H', 'mask_red')
    H2 = cv2.getTrackbarPos('red2_H', 'mask_red')
    S = cv2.getTrackbarPos('red_S', 'mask_red')
    V = cv2.getTrackbarPos('red_V', 'mask_red')
    g_H = cv2.getTrackbarPos('green_H', 'mask_green')
    g_S = cv2.getTrackbarPos('green_S', 'mask_green')
    g_V = cv2.getTrackbarPos('green_V', 'mask_green')
    y_H = cv2.getTrackbarPos('yellow_H', 'mask_yellow')
    y_S = cv2.getTrackbarPos('yellow_S', 'mask_yellow')
    y_V = cv2.getTrackbarPos('yellow_V', 'mask_yellow')
    #red0_low=np.array([H1,S,V])
    #red1_low=np.array([H2,S,V])
    #green_low=np.array([g_H,g_S,g_V])
    #lower_yellow=np.array([y_H,y_S,y_V])
    color_rec(frame)
    t2 = time.time() - t1
    #print(t2)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()