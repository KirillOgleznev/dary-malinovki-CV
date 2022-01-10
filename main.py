from random import randrange

import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

import cv2
import scipy.spatial
import numpy as np

from colorBar import getColor, getBlur, getWatershedSens

COLOR_ACCURACY = -5
#  min_distance чувствительность водораздела
MIN_DISTANCE = 20

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('1.mp4')

img = cv2.imread('data/5.jpg')
(w, h, c) = img.shape
img = cv2.resize(img, (int(h / 4), int(w / 4)))


# (w, h, c) = img.shape
# img = cv2.resize(img, (int(h / 2), int(w / 2)))
# img = img[76:326, 17:691]


# img = cv2.resize(img, (int(h / 2), int(w / 2)))


# Conveyor belt library
# relay = conveyor_lib.Conveyor()


def coords(event, x, y, flags, param):
    if event == 1:
        print(x, y)


def draw_contour(cnt):
    area = cv2.contourArea(cnt)
    ((x1, y1), r1) = cv2.minEnclosingCircle(cnt)

    #  убрать мелкий шум
    if area > 300:
        # cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(belt, ellipse, (0, 0, 255), 2)
        hull = cv2.convexHull(cnt, False)

        # maskTest = cv2.ellipse(np.zeros(img.shape[:2], np.uint8), ellipse, 255, -1)
        # cv2.imshow("Frame2", maskTest)
        areaCnt = int(cv2.contourArea(cnt))
        areaHull = int(cv2.contourArea(hull))

        # qwe = cv2.drawContours(img.copy(), [hull], 0, (0, 0, 255), 6)
        # cv2.drawContours(qwe, [cnt], 0, (0, 255, 0), 2)
        # cv2.imshow("qwe", qwe)
        # cv2.waitKey(0)

        # размер для разделания
        # if areaHull - areaCnt > 3000:
        #     maskTmp = cv2.drawContours(np.zeros(img.shape[:2], np.uint8), [cnt], -1, 255, -1)
        #     res = cv2.bitwise_and(img.copy(), img, mask=maskTmp)
        #
        #     hull2 = cv2.convexHull(cnt, returnPoints=False)
        #
        #     try:
        #         defects = cv2.convexityDefects(cnt, hull2)
        #     except:
        #         print('err')
        #         return
        #
        #     points = []
        #     for i in range(defects.shape[0]):
        #         s, e, f, d = defects[i, 0]
        #         # start = tuple(cnt[s][0])
        #         # end = tuple(cnt[e][0])
        #         far = (int(cnt[f][0]), int(cnt[f][1]))
        #         # cv2.line(res, start, end, [0, 255, 0], 2)
        #         cv2.circle(res, far, 5, [0, 0, 255], -1)
        #         val = cv2.pointPolygonTest(hull, far, True)
        #         if val > 10:
        #             # print(val)
        #             points.append(far)
        #
        #     while len(points) >= 2:
        #         tmp = scipy.spatial.distance.pdist(points)
        #         print(tmp)
        #         minArg = np.argmin(tmp) + 1
        #         d = len(points) - 1
        #         x = 0
        #         while minArg > d:
        #             minArg = minArg - d
        #             d -= 1
        #             x += 1
        #         # cv2.line(frame, points[x], points[x + minArg], (255, 0, 0), 6)
        #         # cv2.drawContours(belt, [hull], 0, (0, 0, 255), 2)
        #
        #         listX = cnt.tolist()
        #         a = listX.index([*points[x]])
        #         b = listX.index([*points[x + minArg]])
        #         if a > b:
        #             a, b = b, a
        #
        #         tmp1 = np.array(cnt[a:b + 1])
        #         # cv2.drawContours(frame, [tmp1], 0, (255, 0, 0), 2)
        #         tmp2 = np.array([*cnt[0:a + 1], *cnt[b:]])
        #         # cv2.drawContours(frame, [tmp2], 0, (0, 0, 255), 2)
        #
        #         del points[x + minArg]
        #         del points[x]
        #
        #         # cv2.imshow("res", res)
        #
        #         draw_contour(tmp1)
        #
        #         draw_contour(tmp2)
        #         return

        # Вывод характеризующего коэфф-а картофеля

        # cv2.drawContours(frame, [hull], 0, (randrange(255), randrange(255), randrange(255)), 2)
        cv2.drawContours(frame, [cnt], 0, (randrange(255), randrange(255), randrange(255)), 2)
        # cv2.ellipse(frame, cv2.fitEllipse(cnt), (0, 0, 255), 2)
        # cv2.circle(frame, (int(x1), int(y1)), int(r1), (randrange(255), randrange(255), 255), 2)

        maskTmp = cv2.drawContours(np.zeros(img.shape[:2], np.uint8), [cnt], -1, 255, -1)
        mean = cv2.mean(img, mask=maskTmp)

        cv2.putText(frame, ('%02d%02d%02d' % mean[:3])[:COLOR_ACCURACY], (int(x1) - 10, int(y1)), 1, 1.2, (0, 0, 0), 6,
                    cv2.LINE_AA)
        cv2.putText(frame, ('%02d%02d%02d' % mean[:3])[:COLOR_ACCURACY], (int(x1) - 10, int(y1)), 1, 1.2, (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        # cv2.putText(frame, str([round(i) for i in mean[:3]]), (x, y), 1, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        # cv2.waitKey(0)
    else:

        cv2.putText(belt, str('Err'), (int(x1) - 10, int(y1)), 1, 1.5, (0, 255, 0), 2, cv2.LINE_AA)


shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

if __name__ == '__main__':
    num = 0
    while True:
        # _, img = cap.read()
        # img = img[303:746, 427:1297]
        # img = img[76:326, 17:691]

        frame = img.copy()
        # cv2.imshow("qwe", img)
        # key = cv2.waitKey(0)

        belt = frame.copy()

        RGB_belt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
        lowHue, lowSat, lowVal, highHue, highSat, highVal = getColor()
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
        threshold = cv2.medianBlur(threshold, getBlur())
        threshold = 255 - threshold

        # D = ndimage.distance_transform_edt(threshold)
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        D = cv2.distanceTransform(threshold, cv2.DIST_L1, 3)
        D = cv2.normalize(D, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # D = ndimage.distance_transform_edt(threshold)
        # cv2.imwrite('png.png', D)

        localMax = peak_local_max(D, indices=False, min_distance=getWatershedSens(), labels=threshold)  # ====================

        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=threshold)  # ====================
        # labels = cv2.watershed(img, markers)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for label in np.unique(labels):
            if label == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            draw_contour(cnt[:, 0])
            # cv2.drawContours(frame, cnts, 0, (randrange(255), randrange(255), randrange(255)), 2)

        # for cnt in contours:
        #     if len(cnt) < 5:
        #         continue
        #     draw_contour(cnt[:, 0])

        cv2.imshow("Frame", frame)
        cv2.imshow("belt", belt)
        cv2.imshow("threshold", threshold)
        cv2.imshow("tmp", hsv_belt)

        # cv2.waitKey(0)

        key = cv2.waitKey(num)
        num = 1
        if key == 27:
            break

    # cap.release()
    cv2.destroyAllWindows()
