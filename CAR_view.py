import cv2
import numpy as np
import serial
import time

# 原始图像
row = 480
cl = 640
# 内圈或外圈
LorR = 1  # 1-外圈，0-内圈

# 循迹图像----------------------
r = 300  # 图像行数
l = 600  # 图像列数

y1l = row - r
y1r = row
x1l = (cl - l) // 2
x1r = cl - ((cl - l) // 2)
# 阈值
hs1 = 120  # 二值化

# 卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
# 循迹采样点
m = 10  # 个数
n = (r) // m  # 间隔
# 图像中心点
xo = l // 2
yo = r // 2

flag_two_line = 0

# //----------------------------

# 颜色识别------------------------
# 各颜色阈值
green_low = np.array([40, 103, 164])
green_up = np.array([77, 255, 255])

red0_low = np.array([5, 166, 224])
red1_low = np.array([160, 166, 224])

red0_up = np.array([10, 255, 255])
red1_up = np.array([180, 255, 255])

yellow_low = np.array([15, 130, 190])  # 黄色低阈值
yellow_up = np.array([34, 255, 255])  # 黄色高阈值

x_for_color = cl // 2

# ------------------------------------------------

# 字符识别---------------------------------
# 字符模板

# --------------------------------------------

#---------特征点匹配模板-----------------

L_sift=cv2.imread('L.jpg')
L_sift=L_sift[0:row,x_for_color+20:cl]
R_sift=cv2.imread('R.jpg')
R_sift=R_sift[0:row,x_for_color+20:cl]

# 停止标志-------------------
flag_A = 0  # 直接停止
flag_B = 0  # 停止5s后启动

th_for_A = 120  # A点判断

t_last = 0  # 上一次识别到标志的时间


def callback(object):
    pass


# 图像处理 获取项目所需图像
def image_change(frame):
    # 循迹图像 dst1
    frame1 = frame[y1l:y1r, x1l:x1r]
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hs1 = cv2.getTrackbarPos('hs1', 'dst1')
    ret1, dst1 = cv2.threshold(gray1, hs1, 255, cv2.THRESH_BINARY)
    dst1 = cv2.dilate(dst1, kernel, iterations=2)
    # # 腐蚀，白区域变小
    dst1 = cv2.erode(dst1, kernel, iterations=1)

    # 颜色识别
    dst2 = frame[40:row - 40, x_for_color:cl]
    # 字符识别
    # dst3 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
    #
    # ret0, dst3 = cv2.threshold(dst3, 93, 255, 1)
    dst3=frame[0:row,x_for_color+20:cl]

    return dst1, dst2, dst3


# 串口通讯-传输巡线位置误差

def ser_write_errox(errox,  ser):
    if errox > 250:
        errox = 250
    elif errox < -250:
        errox = -250
    if errox <= 0:
        errox = -errox
        ser.write('$%d\r'.encode('utf-8') % errox)  # errox为负值
    else:
        ser.write('$%d\n'.encode('utf-8') % errox)  # errox为正值


# 停止标志识别
def flag_rec(dst1, dst2, color):
    global flag_A, flag_B, t_last
    t_now = time.time()
    dt = t_now - t_last
    num1 = np.sum(dst1[r - 10] == 0)
    num1_1 = np.sum(dst1[r - 30] == 0)
    num2 = np.sum(dst1[r - 60] == 0)
    num3 = np.sum(dst1[r - 80] == 0)
    if (num1 >= th_for_A or num1_1 > th_for_A) and num2 != 0 and num3 != 0:
        flag_A = 1
        sift_rec(dst2)  # 字符特征点匹配
        t_last=t_now
        print('A点')
        # ser.write('p'.encode('utf-8'))
    if num1 == 0 and num2 == 0 and num3 == 0:
        flag_B = 1
        print('等停标志')
        # ser.write('s'.encode('utf-8'))
    if dt >= 3:  # 标志清零 3s
        t_last = t_now
        flag_A = 0
        flag_B = 0
    if color == 1:
        print('green')
        # ser.write('d'.encode('utf-8'))
    elif color == 2:
        print('red')
        # ser.write('p'.encode('utf-8'))
    elif color == 3:
        if flag_A == 0:
            print('yellow-stop')
            # ser.write('p'.encode('utf-8'))
    else:
        pass


# 颜色识别  color=1:green ;color=2:red;color=3:yellow;color=0:NULL
def color_rec(dst):
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, green_low, green_up)
    mask_yellow = cv2.inRange(hsv, yellow_low, yellow_up)
    red_mask1 = cv2.inRange(hsv, red0_low, red0_up)
    red_mask2 = cv2.inRange(hsv, red1_low, red1_up)
    mask_red = cv2.bitwise_or(red_mask1, red_mask2)

    contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours1, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy3 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours):
        (x, y, w, h) = cv2.boundingRect(contours[0])
    else:
        w = 0
        h = 0
    if len(contours1):
        (x1, y1, w1, h1) = cv2.boundingRect(contours1[0])
    else:
        w1 = 0
        h1 = 0

    if len(contours2):
        (x2, y2, w2, h2) = cv2.boundingRect(contours2[0])
    else:
        w2 = 0
        h2 = 0

    s1 = w * h
    s2 = w1 * h1
    s3 = w2 * h2
    if s1 >= 20:
        # print("green")
        color = 1  # green
    elif s2 >= 20:
        # print('red')
        color = 2  # red
    elif s3 >= 20:
        # print('yellow')
        color = 3  # yelow
    else:
        color = 0
    return color


# 字符识别-轮廓匹配

# 循迹2.0
def judge_line(pointdata):
    lenth = len(pointdata)
    point_sum = 0
    if lenth > 1:
        for i in range(1, lenth):
            point_sum += 1
            if point_sum > 10 and pointdata[i] - pointdata[i - 1] > 10:
                return i
            elif point_sum == lenth - 1:
                return -1
    else:
        return -2


def line_errox(dst, select_l_r):
    x = np.zeros(m)
    y = np.zeros(m)
    flag_two = 0
    for i in range(0, m):
        index = i * n
        pointsum = np.sum(dst[index] == 0)
        if pointsum == 0:
            x[i] = 0
            y[i] = 0
            continue
        else:
            pointwhere_tuple = np.where(dst[index] == 0)
            pointwhere = np.array(pointwhere_tuple)
            judge = judge_line(pointwhere[0])
            if judge == -2:
                x[i] = 0
                y[i] = 0
            elif judge == -1:
                x[i] = (pointwhere[0][pointsum - 1] + pointwhere[0][0]) / 2
                y[i] = n * (i + 1)
            else:
                if i > 0 and flag_two == 0:
                    for j in range(0, i):
                        x[j] = 0
                        y[j] = 0
                flag_two = 1
                if select_l_r == 1:  # 外圈
                    x[i] = (pointwhere[0][judge] + pointwhere[0][pointsum - 1]) / 2
                else:
                    x[i] = (pointwhere[0][judge - 2] + pointwhere[0][0]) / 2
                y[i] = n * (i + 1)
    x_index = np.nonzero(x)
    x_lenth = len(x_index[0])
    if x_lenth == 0:
        return 0
    x_copy = np.zeros(x_lenth)
    y_copy = np.zeros(x_lenth)
    for i in range(0, x_lenth):
        x_copy[i] = x[x_index[0][i]]
        y_copy[i] = y[x_index[0][i]]
    line_par = np.polyfit(y_copy, x_copy, deg=1)
    line = np.poly1d(line_par)
    print(xo - line(yo))
    point1 = (int(line(0)), 0)
    point2 = (int(line(400)), 400)
    result = cv2.line(dst, point1, point2, (200, 255, 200), 3)
    cv2.imshow('dst1', result)
    return int(xo - line(yo))


# 特征点匹配-----------------------
def sift_rec(img):
    global LorR
    sift = cv2.SIFT_create()
    (kp1, des1) = sift.detectAndCompute(img, None)
    (kp2, des2) = sift.detectAndCompute(R_sift, None)
    (kp3, des3) = sift.detectAndCompute(L_sift, None)
    # print('img 特征点数目：', des1.shape[0])
    # print('R 特征点数目：', des2.shape[0])
    # print('L 特征点数目：', des3.shape[0])
    start = time.time()

    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des1, des2, k=2)
    matches2 = bf.knnMatch(des1, des3, k=2)
    ratio1 = 0.5
    good1 = []
    good2 = []
    for m1, n1 in matches1:
        if m1.distance < ratio1 * n1.distance:
            good1.append([m1])
    for m2, n2 in matches2:
        if m2.distance < ratio1 * n2.distance:
            good2.append([m2])
    print('R匹配数目：', len(good1))
    print('L匹配数目：', len(good2))
    if len(good1) > len(good2):
        LorR = 0
        print('img is R')
    else:
        LorR = 1
        print('img is L')
    end = time.time()

    print('匹配时间:%.5fs' % (end - start))


if __name__ == '__main__':
    # 打开摄像头
    capture = cv2.VideoCapture(1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, cl)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, row)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    cv2.namedWindow('dst1', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('hs1', 'dst1', 90, 255, callback)  # red1
    # 打开硬件串口
    # ser = serial.Serial('/dev/ttyAMA0', 115200)
    # if ser.isOpen == False:
    #     ser.open()

    while True:

        ret, frame = capture.read()  # 摄像头读取图片
        if not ret:
            print("erro:NO camera")
            break
        t_start = time.time()

        # ---------图像处理---------------

        dst1, dst2, dst3 = image_change(frame)

        # --------摄像头循迹--------------

        # errox = find_errox(dst1, m, n, r,thforLorR=10,LorR=LorR)
        errox = line_errox(dst1, select_l_r=LorR)
        # ser_write_errox(errox, ser)

        # ----------颜色识别--------------
        color = color_rec(dst=dst2)
        # -----------字符识别-轮廓------------
        # char_rec(dst3)
        # -----------停止标志识别-&-特征点匹配----
        flag_rec(dst1=dst1, dst2=dst3, color=color)
        t_end = time.time()
        #------------信息输出-----------------
        print('FPS:',int(1 / (t_end-t_start)))
        if flag_A == 1:
            print('flag_A_continue')
            if LorR==1:
                print('L')
            else:
                print('R')
        print("errox:%d" % errox)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()
