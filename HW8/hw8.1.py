import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, count
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        points[count] = x, y
        count += 1
        print(x, y)


count = 0
points = np.zeros((4, 2))
img = cv2.imread('cardDeck.jpg')

while True:

    cv2.imshow('deck', img)
    cv2.setMouseCallback('deck', draw_circle)

    if count == 4:
        height, width = 400, 300
        pts1 = np.float32([points[0], points[1], points[2], points[3]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        cv2.imshow("Out", imgOutput)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()