import time

import cv2
import numpy as np
import skimage.segmentation
from matplotlib import pyplot as plt
from skimage.color import label2rgb

# from blurBar import getBlur as qwe

# icol = (0, 0, 28, 57, 255, 255, 15) # 222.jpg
# icol = (3, 80, 29, 17, 255, 255, 24)  # 111.jpg
# icol = (0, 0, 0, 180, 95, 255, 22)  # 111.jpg inv
# icol = (0, 5, 66, 40, 255, 255, 7)  # 444.jpg
# icol = (9, 87, 24, 45, 184, 255, 29)  # 555.jpg
# icol = (0, 35, 0, 255, 255, 255, 0)
# icol = (0, 0, 3, 255, 255, 255, 2)  # black.jpg
# icol = (0, 10, 54, 33, 255, 255, 7)  # black.jpg
# icol = (1, 3, 66, 94, 118, 255, 1)  # 666.jpg
# icol = (0, 0, 3, 28, 255, 255, 50)  # video.jpg
# icol = (79, 37, 36, 136, 155, 255, 2, 26)  # 3.jpg
icol = (0, 0, 30, 103, 117, 255, 2, 20)  # 5.jpg

def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.resizeWindow('Tracking', 500, 350)
cv2.createTrackbar("l_h", "Tracking", icol[0], 255, nothing)  # создание элемента  Trackbar
cv2.createTrackbar("l_s", "Tracking", icol[1], 255, nothing)
cv2.createTrackbar("l_v", "Tracking", icol[2], 255, nothing)
cv2.createTrackbar("u_h", "Tracking", icol[3], 255, nothing)
cv2.createTrackbar("u_s", "Tracking", icol[4], 255, nothing)
cv2.createTrackbar("u_v", "Tracking", icol[5], 255, nothing)
cv2.createTrackbar("blur", "Tracking", icol[6], 50, nothing)
cv2.createTrackbar("ws_sens", "Tracking", icol[7], 50, nothing)


def getColor():
    lowHue = cv2.getTrackbarPos("l_h", "Tracking")
    lowSat = cv2.getTrackbarPos("l_s", "Tracking")
    lowVal = cv2.getTrackbarPos("l_v", "Tracking")

    highHue = cv2.getTrackbarPos("u_h", "Tracking")
    highSat = cv2.getTrackbarPos("u_s", "Tracking")
    highVal = cv2.getTrackbarPos("u_v", "Tracking")
    return lowHue, lowSat, lowVal, highHue, highSat, highVal


def getWatershedSens():
    sens = cv2.getTrackbarPos("ws_sens", "Tracking")
    return sens


def getBlur():
    blur = cv2.getTrackbarPos("blur", "Tracking")
    return ((blur // 2) * 2) + 1


def coords(event, x, y, flags, param):
    if event == 1:
        print(x, y)


if __name__ == '__main__':
    img = cv2.imread('3.jpg')
    # img = cv2.resize(img, (711, 400))
    # img = img[76:326, 17:691]

    while True:
        timeCheck = time.time()
        # (w, h, c) = frame.shape
        # frame = cv2.resize(frame, (int(h / 3), int(w / 3)))

        hsv_belt = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        colorLow = np.array([78, 40, 23])
        colorHigh = np.array([120, 184, 221])
        threshold1 = cv2.inRange(hsv_belt, colorLow, colorHigh)
        threshold1 = cv2.medianBlur(threshold1, 3)
        frame = cv2.bitwise_and(img, img, mask=255 - threshold1)

        cv2.imshow("threshold1", frame)

        hsv_belt = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        lowHue, lowSat, lowVal, highHue, highSat, highVal = getColor()
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
        threshold = cv2.medianBlur(threshold, getBlur())
        cv2.imshow("threshold", threshold)
        # a, b, c = qwe()
        # a = ((a // 2) * 2) + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.bilateralFilter(gray, 1, 200, 200)
        # blur = cv2.medianBlur(gray, a)

        # laplacian = cv2.Canny(blur, b, c)
        laplacian = cv2.Laplacian(blur, -1, ksize=5, delta=191 - 1000)
        cv2.imshow("laplacian", laplacian)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(laplacian, kernel, iterations=1)
        cv2.imshow("dilation", dilation)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, threshold = cv2.threshold(gray, lowHue, 255, 1)
        # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, contours, 0, (0, 0, 255), 2)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', coords)
        cv2.imshow("image", blur)

        print(60 / (time.time() - timeCheck))
        key = cv2.waitKey(1)
        if key == 27:
            break
