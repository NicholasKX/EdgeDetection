# -*- coding: utf-8 -*-
"""
Created on 2021/10/9 21:13 
@Author: Wu Kaixuan
@File  : Prewitt.py 
@Desc  : Prewitt 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Sobel import Prewitt


img_path = 'lena_512color.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
out_x = Prewitt(img, dx=True)
out_y = Prewitt(img, dy=True)
out_mix = Prewitt(img, dx=True, dy=True)
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Original Image')

plt.subplot(222)
plt.imshow(out_y, 'gray')
plt.title('Prewitt horizontal')

plt.subplot(223)
plt.imshow(out_x, 'gray')
plt.title('Prewitt vertical')

plt.subplot(224)
plt.imshow(out_mix, 'gray')
plt.title('Prewitt mix')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
