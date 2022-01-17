import time

from cv2 import cv2
import numpy as np

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
icol = (0, 0, 33, 102, 85, 255, 2, 20)  # 5.jpg
# icol = (56, 60, 80, 103, 255, 255, 0, 20)  # camera


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
cv2.createTrackbar("blur", "Tracking", icol[6], 10, nothing)
cv2.createTrackbar("ws_sens", "Tracking", icol[7], 300, nothing)


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
    # cap = cv2.VideoCapture(0)

    while True:

        # _, img = cap.read()
        img = cv2.imread('data/5.jpg')
        (w, h, c) = img.shape
        img = cv2.resize(img, (int(h * 0.3), int(w * 0.3)))

        frame = img.copy()

        RGB_belt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
        lowHue, lowSat, lowVal, highHue, highSat, highVal = getColor()
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
        threshold = cv2.medianBlur(threshold, getBlur())
        # threshold = 255 - threshold

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 23)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if len(cnt) < 50:
                continue
            # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('threshold', threshold)

        key = cv2.waitKey(1)
        if key == 27:
            break
