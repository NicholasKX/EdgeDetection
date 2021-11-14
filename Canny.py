# -*- coding: utf-8 -*-
"""
Created on 2021/10/20 9:57 
@Author: Wu Kaixuan
@File  : Canny.py
@Desc  : Canny
"""
import cv2
import numpy as np
from Sobel import Sobel,Prewitt
import matplotlib.pyplot as plt


def Gaussian_filter(img, ksize=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    # Zero padding
    pad = ksize // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
    # 卷积核
    K = np.zeros((ksize, ksize), dtype=np.float32)
    for x in range(-pad, -pad + ksize):  # x,y->[-1,1] 整数
        for y in range(-pad, -pad + ksize):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()  # 归一化

    tmp = out.copy()
    # 滤波
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + ksize, x: x + ksize, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


def Canny(img, th1=60, th2=120):
    # 高斯滤波
    img_gaus = Gaussian_filter(img, sigma=1)
    # 计算梯度 Sobel
    dx = Sobel(img_gaus, dx=True)
    dy = Sobel(img_gaus, dy=True)
    # #求边缘
    edge = Sobel(img_gaus, True, True)
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    # cv2.imshow('edge', edge)
    dx = np.maximum(dx, 1e-10)  # 防止分母为0
    # 梯度向量在点(y,x)处的方向  角度是相对x轴顺时针方向度量的
    angle = np.arctan(dy / dx)
    angle = angle / np.pi * 180  # 弧度-->角度
    angle[angle < -22.5] = 180 + angle[angle < -22.5]  # 整合到0，45，90，135

    # NMS极大值抑制
    H, W = angle.shape
    NMS_edge = edge.copy()
    NMS_edge[0, :] = NMS_edge[H - 1, :] = NMS_edge[:, 0] = NMS_edge[:, W - 1] = 0
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            n1, n2 = None, None
            if (angle[y, x] <= 22.5):
                n1 = NMS_edge[y, x + 1]
                n2 = NMS_edge[y, x - 1]
            elif (angle[y, x] > 22.5 and angle[y, x] <= 67.5):
                n1 = NMS_edge[y - 1, x - 1]
                n2 = NMS_edge[y + 1, x + 1]
            elif (angle[y, x] > 67.5 and angle[y, x] <= 112.5):
                # n1 = NMS_edge[y-1,x]
                # n2 = NMS_edge[y+1,x]
                n1 = NMS_edge[y + 1, x]
                n2 = NMS_edge[y - 1, x]
            elif (angle[y, x] > 112.5 and angle[y, x] <= 157.5):
                n1 = NMS_edge[y + 1, x - 1]
                n2 = NMS_edge[y - 1, x + 1]

            if NMS_edge[y, x] <= n1 or NMS_edge[y, x] <= n2:
                NMS_edge[y, x] = 0

    # 双阈值筛选
    _edge = NMS_edge.copy()
    # 8 - Nearest neighbor
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if NMS_edge[y, x] < th1:
                _edge[y, x] = 0
            elif NMS_edge[y, x] > th2:
                _edge[y, x] = 255
            else:
                if NMS_edge[y - 1, x] > th2 or NMS_edge[y - 1, x - 1] > th2 or NMS_edge[y - 1, x + 1] > th2 or \
                        NMS_edge[y, x - 1] > th2 or NMS_edge[y, x + 1] > th2 or NMS_edge[y + 1, x - 1] or \
                        NMS_edge[y + 1, x] > th2 or NMS_edge[y + 1, x + 1] > th2:
                    _edge[y, x] = 255

    return _edge

if __name__=="__main__":
    img_path = 'lena_512color.jpg'
    img = cv2.imread(img_path)
    _img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    import time
    start = time.time()
    img2 = Canny(img, 100, 120)
    cv2.imwrite('lena_canny.jpg',img2)
    end = time.time()
    print('1',end-start)
    start1 = time.time()
    img_sobel =Sobel(img,dx=1,dy=1)
    end1 = time.time()
    print('2',end1-start1)
    start2 = time.time()
    img_pre = Prewitt(img,dx=1,dy=1)
    end2 = time.time()
    print('3',end2-start2)
    # img3 = cv2.Canny(img, 120, 140)  # 120 140
    # # cv2.imshow('1',img1)
    # cv2.imshow('2', img2)
    # cv2.imshow('3', img3)
    # cv2.waitKey()
    plt.figure(figsize=(8, 5))
    plt.subplot(221)
    plt.imshow(_img,'gray')
    plt.title('Original Image')

    plt.subplot(222)
    plt.imshow(img2, 'gray')
    plt.title('Canny')

    plt.subplot(223)
    plt.imshow(img_sobel, 'gray')
    plt.title('Sobel')

    plt.subplot(224)
    plt.imshow(img_pre, 'gray')
    plt.title('Prewitt')

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()
