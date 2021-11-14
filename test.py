# -*- coding: utf-8 -*-
"""
Created on 2021/10/11 17:12 
@Author: Wu Kaixuan
@File  : test.py 
@Desc  : test 
"""
import numpy as np
import cv2
# img_path = 'lena_512color.jpg'
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_gaus = cv2.GaussianBlur(img,ksize=(3,3),sigmaX=1.5,sigmaY=1.5)
# cv2.imshow('dst',img_gaus)
# cv2.imwrite('gauss.jpg',img_gaus)
# cv2.waitKey()
def Gaussian_filter(ksize=3, sigma=1.3):
    # if len(img.shape) == 3:
    #     H, W, C = img.shape
    # else:
    #     img = np.expand_dims(img, axis=-1)
    #     H, W, C = img.shape
    # # Zero padding
    # pad = ksize // 2
    # out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
    # out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
    # 卷积核
    pad=1
    K = np.zeros((ksize, ksize), dtype=np.float32)
    for x in range(-pad, -pad + ksize):  # x,y->[-1,1] 整数
        for y in range(-pad, -pad + ksize):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()  # 归一化
    return K.astype(np.float32)
# print(Gaussian_filter(sigma=1))
from Sobel import BGR2GRAY
img_path = 'lena_512color.jpg'
img = cv2.imread(img_path)
res = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res1 = BGR2GRAY(img)
# res = cv2.GaussianBlur(img,ksize=(3,3),sigmaX=2,sigmaY=2)
cv2.imwrite('lena_gray.jpg',res1)
cv2.imshow('res',res)
cv2.imshow('res1',res1)
cv2.waitKey()



