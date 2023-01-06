import cv2
import numpy as np
import time

# 图像大小控制
r = 400
l = 640

row = 480
cl = 640

x1 = (640 - l) // 2
x2 = 640 - ((640 - l) // 2)

# 图像中心
xo = l // 2
yo = r - 80

# 采样间距 10个采样点
m = 10
n = int(r // m)

# y = int(n - r)

# 防大色块阴影干扰
ms = 5
# 阈值
hs = 110

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cl)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, row)
L = cv2.imread('L.jpg', 1)
L = cv2.resize(L, (640, 400))
# cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('dst', cv2.WINDOW_AUTOSIZE)


def judge_line(pointdata):
    lenth = len(pointdata)
    # print('lenth:{}'.format(lenth))
    # print(pointdata)
    point_sum = 0
    if lenth > 1:
        for i in range(1, lenth):
            point_sum += 1
            if point_sum > 20 and pointdata[i] - pointdata[i - 1] > 10:
                #print(pointdata[i])
                return i
            elif point_sum == lenth - 1:
                return -1
    else:
        return -2


def line_errox(dst, select_l_r):
    x = np.zeros(m)
    y = np.zeros(m)
    for i in range(0, m):
        index = r - 1 - i * n
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
                if select_l_r == 1:  # 外圈
                    x[i] = (pointwhere[0][judge - 1] + pointwhere[0][pointsum - 1]) / 2
                else:
                    x[i] = (pointwhere[0][judge - 2] + pointwhere[0][0]) / 2
                y[i] = n * (i + 1)
    x_index = np.nonzero(x)
    #x_index = np.array(x_index)
    x_lenth = len(x_index[0])
    x_copy = np.zeros(x_lenth)
    y_copy = np.zeros(x_lenth)
    for i in range(0, x_lenth):
        x_copy[i] = x[x_index[0][i]]
        y_copy[i] = y[x_index[0][i]]
    # print('x_copy:{}'.format(x_copy))
    # print('y_copy:{}'.format(y_copy))
    line_par = np.polyfit(y_copy, x_copy, deg=1)
    line = np.poly1d(line_par)
    print(xo-line(r-10))
    point1 = (int(line(0)), 0)
    point2 = (int(line(400)), 400)
    result = cv2.line(dst, point1, point2, (200, 255, 200), 3)
    cv2.imshow('dst', result)
    return int(xo-line(r-10))





while True:
    ret, frame = capture.read()  # 摄像头读取图片
    t1 = time.time()
    frame = frame[480 - r:480, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    ret2, dst = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    dst = cv2.dilate(dst, None, iterations=2)
    # # 腐蚀，白区域变小
    dst = cv2.erode(dst, None, iterations=1)
    errx = line_errox(dst, 1)
    t2 = time.time()
    print(int(1 / (t2 - t1)))
    # print("errox:%d" % errx)
    # cv2.imshow('dst', dst)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
