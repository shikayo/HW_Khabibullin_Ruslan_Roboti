import cv2
import numpy as np

points = np.zeros((100, 2), np.int32)
count = 0
distPix = 0

def draw_line(event, x, y, flags, params):
    global mouseX, mouseY, count
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(map_route, (x, y), 3, (255, 0, 0), -1)
        points[count] = x, y
        count += 1
        print(x,y)
        if count >= 2:
            cv2.line(map_route, points[count - 2], points[count - 1], (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)


map_route = cv2.imread('map.png')

while True:

    cv2.imshow('map', map_route)
    cv2.setMouseCallback('map', draw_line)

    if cv2.waitKey(1) & 0xFF == 27:
        if count >= 2:
            cv2.imwrite('resultMap.png', map_route)
        break
cv2.destroyAllWindows()