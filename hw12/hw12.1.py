import cv2
import numpy as np
import cv2 as cv
import sys

from cv2.gapi import kernel
from matplotlib import pyplot as plt


def sobel():
    window_name = ('S')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    # Load the image
    src = cv.imread('obj/coin.jpg', cv.IMREAD_COLOR)
    # Check if image is loaded fine
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    plt.figure(figsize=(15, 15))
    plt.imshow(grad)
    plt.title('S')
    plt.xticks([]), plt.yticks([])
    plt.show()
    #     cv.imshow(window_name, grad)
    #     cv.waitKey(0)
    return 0


sobel()

img = cv.imread('obj/coin.jpg')
canny = cv.Canny(img, 100, 200)

blur = cv.blur(img, (5, 5))
plt.figure(figsize=(15, 15))
plt.imshow(blur)
plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

dilate = cv.dilate(canny, kernel=kernel, iterations=1)
plt.figure(figsize=(15, 15))
plt.imshow(dilate, cmap='gray')
plt.title('dilate')
plt.show()

erosion = cv2.erode(dilate, kernel=kernel, iterations=1)
plt.figure(figsize=(15, 15))
plt.imshow(erosion, cmap='gray')
plt.title('erosion')
plt.show()