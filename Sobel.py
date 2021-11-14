# -*- coding: utf-8 -*-
"""
Created on 2021/10/9 20:53 
@Author: Wu Kaixuan
@File  : Sobel.py 
@Desc  : Sobel 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    out = 0.299 * r + 0.578 * g + 0.114 * b
    out = out.astype(np.uint8)
    return out

def general_filter(img, Gx, Gy, ksize=3):
    H,W = img.shape[:2]
    img_gray=BGR2GRAY(img)
    # Zero padding
    pad = ksize // 2
    img_padding = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
    img_padding[pad: pad + H, pad: pad + W] = img_gray.copy().astype(np.float32)
    # computing
    out_x = img_padding.copy()
    out_y = img_padding.copy()
    tmp = img_padding.copy()
    out_mix = img_padding.copy()
    # start = time.time()
    for y in range(H):
        for x in range(W):
            out_y[pad + y, pad + x] = np.sum(Gy * (tmp[y: y + ksize, x: x + ksize]))
            out_x[pad + y, pad + x] = np.sum(Gx * (tmp[y: y + ksize, x: x + ksize]))
            # out_mix[pad + y, pad + x] = abs(out_y[pad + y, pad + x]) + abs(out_x[pad + y, pad + x])
            out_mix[pad + y, pad + x] = math.sqrt(pow(out_y[pad + y, pad + x],2) + pow(out_x[pad + y, pad + x],2))
    # end=time.time()
    # print('time',end-start)
    out_x = np.clip(out_x, 0, 255)
    out_y = np.clip(out_y, 0, 255)
    out_mix = np.clip(out_mix, 0, 255).astype(np.uint8)
    # print(out_mix)

    out_x = out_x[pad: pad + H, pad: pad + W].astype(np.uint8)
    # out_x[0,:] = out_x[H - 1, :] = out_x[:, 0] = out_x[:, W - 1] = 0
    out_y = out_y[pad: pad + H, pad: pad + W].astype(np.uint8)
    # out_y[0, :] = out_y[H - 1, :] = out_y[:, 0] = out_y[:, W - 1] = 0
    out_mix = out_mix[pad: pad + H, pad: pad + W].astype(np.uint8)
    # out_mix[0, :] = out_mix[H - 1, :] = out_mix[:, 0] = out_mix[:, W - 1] = 0

    return out_x, out_y, out_mix
def Sobel(img, dx=False, dy=False, ksize=3):
    # Sobel horizontal
    Gx = [[-1., 0., 1.],
          [-2., 0., 2.],
          [-1., 0., 1.]]
    # Soebl vertical
    Gy = [[-1., -2., -1.],
          [0., 0., 0.],
          [1., 2., 1.]]

    out_x, out_y, out_mix = general_filter(img, Gx, Gy,ksize)
    if dx and dy == False:
        return out_x
    elif dy and dx == False:
        return out_y
    elif dx and dy:
        return out_mix
    else:
        return img
def Prewitt(img,dx=False, dy=False, ksize=3):
    # Sobel horizontal
    Gx = [[-1., 0., 1.],
          [-1., 0., 1.],
          [-1., 0., 1.]]
    # Soebl vertical
    Gy = [[-1., -1., -1.],
          [0., 0., 0.],
          [1., 1., 1.]]
    out_x, out_y, out_mix = general_filter(img, Gx, Gy, ksize)
    if dx and dy == False:
        return out_x
    elif dy and dx == False:
        return out_y
    elif dx and dy:
        return out_mix
    else:
        return img


if __name__=="__main__":
    img_path = 'lena_512color.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_x = Sobel(img, dx=True)
    out_y = Sobel(img, dy=True)
    out_mix = Sobel(img, dx=True, dy=True)
    out_mix2 = Prewitt(img,dx=True,dy=True)
    # guanfang = cv2.Sobel(img,ddepth=cv2.CV_8U,dx=1,dy=1)
    # cv2.imshow('guan',guanfang)
    # cv2.waitKey()
    cv2.imwrite('lena_sobel.jpg',out_mix)
    cv2.imwrite('lena_pre.jpg',out_mix2)
    # plt.subplot(221)
    # plt.imshow(img_rgb)
    # plt.title('Original Image')
    #
    # plt.subplot(222)
    # plt.imshow(out_y, 'gray')
    # plt.title('Sobel horizontal')
    #
    # plt.subplot(223)
    # plt.imshow(out_x, 'gray')
    # plt.title('Sobel vertical')
    #
    # plt.subplot(224)
    # plt.imshow(out_mix, 'gray')
    # plt.title('Sobel mix')
    #
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.show()
