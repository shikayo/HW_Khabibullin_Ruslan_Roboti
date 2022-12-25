import random
import cv2


def add_noise(img, percentage):
    row, col = img.shape

    number_of_pixels = row * col
    perc = percentage / 100
    for i in range(round(round(number_of_pixels * perc) / 4)):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    for i in range(round(round(number_of_pixels * perc) / 4)):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def on_change(value):
    global noise
    print(value)
    noise = value
    salted = add_noise(img, value)
    cv2.imshow('salt-pepper noises', salted)
    cv2.waitKey(1)


img = cv2.imread('Genshin-Impact-anime.jpg',
                 cv2.IMREAD_GRAYSCALE)
windowName = 'salt-pepper noises'
cv2.imshow(windowName, add_noise(img, 0))

cv2.createTrackbar('noise:', windowName, 0, 100, on_change)

cv2.waitKey(0)
cv2.destroyAllWindows()