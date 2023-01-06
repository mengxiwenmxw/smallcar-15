import cv2
import numpy as np
import time
img=cv2.imread('L_R/img5.jpg')
img_wh=img.shape
print(img_wh)
R=cv2.imread('L_R/img0.jpg')
L=cv2.imread('L_R/img7.jpg')
img=img[0:img_wh[0],img_wh[1]//2+20:img_wh[1]]
R=R[0:img_wh[0],img_wh[1]//2+20:img_wh[1]]
L=L[0:img_wh[0],img_wh[1]//2+20:img_wh[1]]
# sift=cv2.SIFT_create()
# (kp1,des1)=sift.detectAndCompute(img, None)
# (kp2,des2)=sift.detectAndCompute(R, None)
# (kp3,des3)=sift.detectAndCompute(L, None)
# print('img 特征点数目：',des1.shape[0])
# print('R 特征点数目：',des2.shape[0])
# print('L 特征点数目：',des3.shape[0])
# sift_img=cv2.drawKeypoints(img,kp1,img,color=(200,255,200))
# sift_R=cv2.drawKeypoints(R,kp2,R,color=(200,255,200))
# sift_cat1=np.hstack((sift_img,sift_R))
# sift_R=cv2.drawKeypoints(L,kp3,L,color=(200,255,200))
# #cv2.imshow('sift_cat1',sift_cat1)
# #cv2.waitKey()
#
# start=time.time()
#
# bf=cv2.BFMatcher()
# matches1=bf.knnMatch(des1,des2,k=2)
# matches2=bf.knnMatch(des1,des3,k=2)
# ratio1=0.5
# good1=[]
# good2=[]
# for m1,n1 in matches1:
#     if m1.distance<ratio1*n1.distance:
#         good1.append([m1])
# for m2,n2 in matches2:
#     if m2.distance<ratio1*n2.distance:
#         good2.append([m2])
# print('R匹配数目：',len(good1))
# print('L匹配数目：',len(good2))
# end=time.time()
#
# print('匹配时间:%.5fs'%(end-start))
# match_result_R=cv2.drawMatchesKnn(img,kp1,R,kp2,good1,None,flags=2)
# match_result_L=cv2.drawMatchesKnn(img,kp1,L,kp3,good2,None,flags=2)
# cv2.imshow('result_R',match_result_R)
# cv2.imshow('result_L',match_result_L)
# cv2.waitKey()

def sift_rec(img):
    sift = cv2.SIFT_create()
    (kp1, des1) = sift.detectAndCompute(img, None)
    (kp2, des2) = sift.detectAndCompute(R, None)
    (kp3, des3) = sift.detectAndCompute(L, None)
    print('img 特征点数目：', des1.shape[0])
    print('R 特征点数目：', des2.shape[0])
    print('L 特征点数目：', des3.shape[0])
    start = time.time()

    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des1, des2, k=2)
    matches2 = bf.knnMatch(des1, des3, k=2)
    ratio1 = 0.6
    good1 = []
    good2 = []
    for m1, n1 in matches1:
        if m1.distance < ratio1 * n1.distance:
            good1.append([m1])
    for m2, n2 in matches2:
        if m2.distance < ratio1 * n2.distance:
            good2.append([m2])
    #print('R匹配数目：', len(good1))
    #print('L匹配数目：', len(good2))
    if len(good1)>len(good2):
        print('img is R')
    else:
        print('img is L')
    end = time.time()

    print('匹配时间:%.5fs' % (end - start))
    match_result_R = cv2.drawMatchesKnn(img, kp1, R, kp2, good1, None, flags=2)
    match_result_L = cv2.drawMatchesKnn(img, kp1, L, kp3, good2, None, flags=2)
    cv2.imshow('result_R', match_result_R)
    cv2.imshow('result_L', match_result_L)
    cv2.waitKey()

sift_rec(img)