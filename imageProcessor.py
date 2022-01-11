from random import randrange
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from cv2 import cv2

import numpy as np

from colorBar import getColor, getBlur, getWatershedSens


class ImageProcessor(object):
    COLOR_ACCURACY = -5
    potato_id = 0
    potatoes = []

    def __init__(self, src):
        # cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('1.mp4')
        self.img = cv2.imread(src)
        self.frame = self.img.copy()
        self.hsv_belt = None
        self.threshold = None
        self.belt = None
        # self.shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    def resize(self, num):
        (w, h, c) = self.img.shape
        # self.img = cv2.resize(self.img, (int(h / 4), int(w / 4)))
        self.frame = cv2.resize(self.frame, (int(h / num), int(w / num)))
        self.threshold = cv2.resize(self.threshold, (int(h / num), int(w / num)))
        self.hsv_belt = cv2.resize(self.hsv_belt, (int(h / num), int(w / num)))
        self.belt = cv2.resize(self.belt, (int(h / num), int(w / num)))
        # img = img[303:746, 427:1297]
        # img = img[76:326, 17:691]

    def showAll(self):
        cv2.imshow("Frame", self.frame)
        cv2.imshow("belt", self.belt)
        cv2.imshow("threshold", self.threshold)
        cv2.imshow("tmp", self.hsv_belt)

    def get_key(self, num):
        key = cv2.waitKey(num)
        return key

    def watershed(self):
        RGB_belt = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
        lowHue, lowSat, lowVal, highHue, highSat, highVal = getColor()
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        self.threshold = cv2.inRange(self.hsv_belt, colorLow, colorHigh)
        self.threshold = cv2.medianBlur(self.threshold, getBlur())
        self.threshold = 255 - self.threshold
        closing = cv2.morphologyEx(self.threshold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        D = cv2.distanceTransform(self.threshold, cv2.DIST_L1, 3)
        D = cv2.normalize(D, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imwrite('png.png', D)

        localMax = peak_local_max(D, indices=False, min_distance=getWatershedSens(),
                                  labels=self.threshold)

        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        self.labels = watershed(-D, markers, mask=self.threshold)

    def find_and_draw_contours(self):
        # _, img = cap.read()
        self.frame = self.img.copy()
        self.belt = self.frame.copy()
        self.watershed()
        for label in np.unique(self.labels):
            if label == 0:
                continue
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[self.labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            self.draw_contour(cnt[:, 0])
            # cv2.drawContours(frame, cnts, 0, (randrange(255), randrange(255), randrange(255)), 2)

    def draw_contour(self, cntr):
        area = cv2.contourArea(cntr)
        ((x1, y1), r1) = cv2.minEnclosingCircle(cntr)

        #  убрать мелкий шум
        if area > 3000:
            # cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # ellipse = cv2.fitEllipse(cnt)
            # cv2.ellipse(belt, ellipse, (0, 0, 255), 2)
            hull = cv2.convexHull(cntr, False)

            # maskTest = cv2.ellipse(np.zeros(img.shape[:2], np.uint8), ellipse, 255, -1)
            # cv2.imshow("Frame2", maskTest)
            areaCnt = int(cv2.contourArea(cntr))
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
            cv2.drawContours(self.frame, [cntr], 0, (randrange(255), randrange(255), randrange(255)), 10)
            # cv2.ellipse(frame, cv2.fitEllipse(cnt), (0, 0, 255), 2)
            # cv2.circle(frame, (int(x1), int(y1)), int(r1), (randrange(255), randrange(255), 255), 2)

            maskTmp = cv2.drawContours(np.zeros(self.img.shape[:2], np.uint8), [cntr], -1, 255, -1)
            mean = cv2.mean(self.img, mask=maskTmp)
            self.potatoes.append([self.potato_id, [int(x) for x in mean]])
            # cv2.putText(self.frame, ('%02d%02d%02d' % mean[:3])[:self.COLOR_ACCURACY], (int(x1) - 50, int(y1) + 20),
            #             1, 6, (0, 255, 0), 6, cv2.LINE_AA)
            cv2.putText(self.frame, str(self.potato_id), (int(x1) - 50, int(y1) + 20), 1, 6, (0, 255, 0), 6, cv2.LINE_AA)
            # cv2.putText(frame, str([round(i) for i in mean[:3]]), (x, y), 1, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            self.potato_id += 1
        else:

            cv2.putText(self.belt, str('Err'), (int(x1) - 10, int(y1)), 1, 6, (0, 255, 0), 6, cv2.LINE_AA)

    def create_report(self):
        # print(self.potatoes)
        f = open('text.txt', 'w')
        for i in self.potatoes:
            f.write(str(i)[1:-1] + '\n')
        f.close()
