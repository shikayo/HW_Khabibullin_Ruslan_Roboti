import cv
import matplotlib.pyplot as plt

default_file = 'map1.pgm'
src = cv.imread(cv.samples.findFile(default_file), cv.IMREAD_GRAYSCALE)
gray_image = cv.Canny(src, 50, 200, None, 3)
cdst = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
plt.figure(figsize=(15,15))
plt.imshow(cdst)
plt.title("map")