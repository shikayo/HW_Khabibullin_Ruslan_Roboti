import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../Picture/1556708032_1.jpg')
scale_up_x = 0.5
scale_up_y = 0.5
scale_down = 0.6
width = int(img.shape[1] * scale_up_x)
height = int(img.shape[0] * scale_up_y)
dim = (width, height)
a = res_inter_nearset = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
b = res_inter_linear = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
c = res_inter_area = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(a)
print(b)
print(c)

cv2.imshow('INTER_NEARES',a)
cv2.waitKey()

cv2.imshow('INTER_LINEAR',b)
cv2.waitKey()

cv2.imshow('INTER_AREA',c)
cv2.waitKey()
cv2.destroyAllWindows()

size = img.shape
print(size)

columns = 3
rows = 3

y_patch_length = size[0] / 3
x_patch_length = size[1] / 3
for i in range(0, rows):
    for j in range(0, columns):
        int_x_patch = int(x_patch_length)
        int_y_patch = int(y_patch_length)
        cropped_x = int_x_patch * j
        cropped_x1 = int_x_patch * (j+1)
        cropped_y = int_y_patch * i
        cropped_y1 = int_y_patch * (i+1)
        patch = img[cropped_y : cropped_y1, cropped_x : cropped_x1]
        cv2.imwrite('../Picture/patch('+ str(i) +':'+ str(j) + '.png', patch)

import matplotlib.pyplot as plt

fig1, fig2 = plt.subplots(nrows=rows, ncols=columns)
for i in range(0, rows):
    for j in range(0, columns):
        img = plt.imread('../Picture/patch('+ str(i) +':'+ str(j) + '.png')
        fig2[i,j].axis('off')
        fig2[i,j].imshow(img)
plt.tight_layout()
plt.show()

