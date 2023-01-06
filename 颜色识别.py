import cv2
import numpy as np
import time
font = cv2.FONT_HERSHEY_SIMPLEX
s = 1000
def callback(object):
    pass

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow('mask_yellow', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('mask_green', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('mask_red', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('yellow_low','mask_yellow',200,255,callback)
cv2.createTrackbar('red_low','mask_red',200,255,callback)
cv2.createTrackbar('green_low','mask_green',200,255,callback)
# 各颜色阈值
# green_low = np.array([35, 43, 46])
# green_up = np.array([77, 255, 255])
#
# red0_low = np.array([0, 43, 46])
# red1_low = np.array([156, 43, 46])
#
# red0_up = np.array([10, 255, 255])
# red1_up = np.array([180, 255, 255])
#
# lower_yellow = np.array([11, 43, 46])  # 黄色低阈值
# upper_yellow = np.array([34, 255, 255])  # 黄色高阈值
green_low = np.array([35, 43, 200])
green_up = np.array([77, 255, 255])

red0_low = np.array([0, 43, 200])
red1_low = np.array([156, 43, 46])

red0_up = np.array([10, 255, 255])
red1_up = np.array([180, 255, 255])

lower_yellow = np.array([11, 43, 200])  # 黄色低阈值
upper_yellow = np.array([34, 255, 255])  # 黄色高阈值


def CV_RGB():
    while True:
        ret, frame = capture.read()  # 摄像头读取图片
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, green_low, green_up)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        red_mask1 = cv2.inRange(hsv, red0_low, red0_up)
        red_mask2 = cv2.inRange(hsv, red1_low, red1_up)
        mask_red = cv2.bitwise_or(red_mask1, red_mask2)
        # 中值滤波
        # mask_green = cv2.medianBlur(mask_green, 7)
        # mask_red=cv2.medianBlur(mask_red, 7)
        # mask_yellow = cv2.medianBlur(mask_yellow, 7)
        # mask=cv2.bitwise_or(mask_red,mask_yellow,mask_green)

        contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours2, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours3, hierarchy3 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w * h > s:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Green", (x, y - 5), font, 0.7, (0, 255, 0), 2)

        for cnt2 in contours2:
            (x2, y2, w2, h2) = cv2.boundingRect(cnt2)
            if w2 * h2 > s:
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                cv2.putText(frame, "Red", (x2, y2 - 5), font, 0.7, (0, 0, 255), 2)

        for cnt3 in contours3:
            (x3, y3, w3, h3) = cv2.boundingRect(cnt3)
            if w3 * h3 > s:
                cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 255), 2)
                cv2.putText(frame, "Yellow", (x3, y3 - 5), font, 0.7, (0, 255, 255), 2)

        # cv2.imshow("mask_green",mask_green)
        # cv2.imshow("mask_red", mask_red)
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

#retunr color=0:green ; color=1:red ; color=2:yeloow
def color_rec(dst):
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
    if s1>=20:
        print("green")
        color=0    #green
    elif s2>=20:
        print('red')
        color=1    #red
    elif s3>=20:
        print('yellow')
        color=2    #yelow
    cv2.imshow("mask_red", mask_red)
    cv2.imshow("mask_yellow", mask_yellow)
    cv2.imshow("mask_green", mask_green)
    return color

while True:
    ret, frame = capture.read()
    t1=time.time()
    value1=cv2.getTrackbarPos('yellow_low','mask_yellow')
    value2=cv2.getTrackbarPos('red_low', 'mask_red')
    value3=cv2.getTrackbarPos('green_low', 'mask_green')
    lower_yellow = np.array([11, 43, value1])
    red0_low = np.array([0, 43, value2])
    red1_low = np.array([156, 43, value2])
    green_low = np.array([35, 43, value3])
    color_rec(frame)
    t2=time.time()-t1
    #print(t2)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
