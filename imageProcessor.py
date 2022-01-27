import time

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from cv2 import cv2

import numpy as np

# from colorBar import getColor, getBlur, getWatershedSens
from colorBar import TrackingBar


class ImageProcessor:
    COLOR_ACCURACY = -5
    # Множитель преобразования пикселей в сантиметры
    PIXEL_TO_CM = 0.1029
    COLUMNS_NAMES = ['id', 'area_cm', 'RGB', 'variety', 'ellipsoid_shape',
                     'weight', 'weight_fraction', 'min_diameter_mm', 'size_fraction']
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters_create()

    def __init__(self, srcImg=None, srcVideo=None, camera=0, ratio=1.0, img=None):
        """

        :param srcImg: Путь до изображения
        :param srcVideo: Путь до видео
        :param camera: Номер камеры подключенной к ПК
        :param ratio: Множитель изменения входящего изображения (можно использовать для оптимизации)
        """
        self.pointsQR = []
        self.ratio = ratio
        if img:
            self.img = img
            self.cap = None
        elif srcImg:
            self.img = cv2.imread(srcImg)
            (w, h, c) = self.img.shape
            self.img = cv2.resize(self.img, (int(h * ratio), int(w * ratio)))
            self.cap = None
        elif srcVideo:
            self.cap = cv2.VideoCapture(srcVideo)
            self.update_frame()
        else:
            self.cap = cv2.VideoCapture(camera)
            self.update_frame()
        if self.img is None:
            raise Exception('Scr not found!')
        self.frame = self.img.copy()
        self.potato_id = 0
        self.potatoes = []
        self.hsv_belt = None
        self.threshold = None
        self.belt = self.frame.copy()
        # self.shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    def getFrame(self):
        return self.frame

    def qr_code_detector(self):
        self.pointsQR = []
        det = cv2.QRCodeDetector()
        # ret, cnt = det.detectMulti(np.hstack([self.img, self.img]))
        decodedText, points, _ = det.detectAndDecode(self.img)
        print(len(decodedText), decodedText)
        if len(decodedText) < 5:
            return
        if points is not None:
            cnt = []
            points = points[0]
            for i in range(len(points)):
                pt1 = [int(val) for val in points[i]]
                cnt.append(pt1)
            xc = sum([cnt[0][0], cnt[1][0], cnt[2][0], cnt[3][0]]) / 4
            yc = sum([cnt[0][1], cnt[1][1], cnt[2][1], cnt[3][1]]) / 4
            # cv2.circle(self.img, (int(xc), int(yc)), 4, (0, 0, 255), -1)
            cv2.drawContours(self.img, [np.array(cnt)], 0, (0, 0, 255), -1)
            cv2.drawContours(self.img, [np.array(cnt)], 0, (0, 0, 255), 3)

            # a = cv2.pointPolygonTest(np.array(cnt), (int(xc), int(yc)), False)
            self.pointsQR.append([(int(xc), int(yc)), decodedText])

    def aruco_marker(self):
        cnt, _, _ = cv2.aruco.detectMarkers(self.img, self.aruco_dict, parameters=self.parameters)
        int_corners = np.int0(cnt)
        cv2.polylines(self.belt, int_corners, True, (0, 255, 0), 2)
        if len(cnt) != 0:
            aruco_perimeter = cv2.arcLength(cnt[0], True)
            pixel_cm_ratio = 40 / aruco_perimeter
            (x, y), (w, h), angle = cv2.minAreaRect(cnt[0])
            cv2.putText(self.belt, "Width {} cm".format(round(w * pixel_cm_ratio, 1)),
                        (int(x - 80), int(y - 20)),
                        5, 1, (250, 0, 250), 1)
            cv2.putText(self.belt, "Height {} cm".format(round(h * pixel_cm_ratio, 1)),
                        (int(x - 80), int(y)),
                        5, 1, (250, 0, 250), 1)
            self.PIXEL_TO_CM = pixel_cm_ratio
            mask_image = cv2.drawContours(np.zeros(self.img.shape[:2], np.uint8), int_corners, -1, 255, -1)
            border_points = np.array(np.where(mask_image == 255)).transpose()
            for point in border_points:
                self.img[point[0], point[1]] = [50, 150, 50]

    def update_frame(self):
        """
                Обновляет кадр (Только для видео)
                :return: None
                """
        if self.cap:
            _, self.img = self.cap.read()
            if self.img:
                (w, h, c) = self.img.shape
                self.img = cv2.resize(self.img, (int(h * self.ratio), int(w * self.ratio)))

    def resize(self, num):
        """
        Метод изменения размера окана
        :param num: Коэффициент уменьшения (Множитель)
        :return: None
        """
        (w, h, c) = self.img.shape
        # self.img = cv2.resize(self.img, (int(h / 4), int(w / 4)))
        self.frame = cv2.resize(self.frame, (int(h * num), int(w * num)))
        self.threshold = cv2.resize(self.threshold, (int(h * num), int(w * num)))
        self.hsv_belt = cv2.resize(self.hsv_belt, (int(h * num), int(w * num)))
        self.belt = cv2.resize(self.belt, (int(h * num), int(w * num)))
        # img = img[303:746, 427:1297]
        # img = img[76:326, 17:691]

    def showAll(self, frame=True, belt=True, threshold=True):
        """
        Метод вывода информационных окон
        :return: None
        """
        if frame:
            cv2.imshow("Frame", self.frame)
        if belt:
            cv2.imshow("Orig", self.belt)
        if threshold:
            cv2.imshow("Threshold", self.threshold)

    def get_key(self, num):
        """
        Метод для вызова паузы и считывания введенного символа
        :param num: При значении 0 ожидает ввода люього символа
        :return: Символ который был введен
        """
        key = cv2.waitKey(num)
        return key

    def watershed(self, data=None):
        """
        Алгоритм сегментации близко находящихся клубней
        :return: Разделенный контур
        """
        RGB_belt = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
        if data is None:
            lowHue, lowSat, lowVal, highHue, highSat, highVal, blur, watershedSens = TrackingBar.icol
        else:
            from colorBar import icol
            lowHue, lowSat, lowVal, highHue, highSat, highVal = icol[:6]
            blur = ((icol[6] // 2) * 2) + 1
            watershedSens = icol[7]
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        self.threshold = cv2.inRange(self.hsv_belt, colorLow, colorHigh)
        self.threshold = cv2.medianBlur(self.threshold, blur)
        self.threshold = 255 - self.threshold
        # closing = cv2.morphologyEx(self.threshold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        # D = cv2.distanceTransform(self.threshold, cv2.DIST_L1, 3)
        # D = cv2.normalize(D, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        D = ndimage.distance_transform_edt(self.threshold)
        # cv2.imwrite('png.png', D)

        # localMax = peak_local_max(D, indices=False, min_distance=watershedSens,
        #                           labels=self.threshold)
        # markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        coords = peak_local_max(D, min_distance=watershedSens, labels=self.threshold)
        mask = np.zeros(D.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)

        return watershed(-D, markers, mask=self.threshold)

    def find_and_draw_contours(self, icol=None):
        """
        Метод находит контуры картофеля и рисует их на окне "Frame"
        :return: None
        """
        # _, img = cap.read()
        self.potato_id = 0
        self.potatoes = [self.COLUMNS_NAMES]
        self.frame = self.img.copy()
        self.belt = self.img.copy()
        # self.belt = self.frame.copy()
        if icol is None:
            labels = self.watershed()
        else:
            labels = self.watershed(icol)
        for label in np.unique(labels):
            if label == 0:
                continue
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            self.draw_contour(cnt[:, 0])
            # cv2.drawContours(frame, cnts, 0, (randrange(255), randrange(255), randrange(255)), 2)

    def draw_contour(self, cntr):
        """
        Рисует контур картофеля на окне "Frame"
           Также вызывает метод анализа клубня
        :param cntr: Контур 1 клубня
        :return: None
        """
        area = cv2.contourArea(cntr)
        ((x1, y1), r1) = cv2.minEnclosingCircle(cntr)
        cv2.circle(self.belt, (int(x1), int(y1)), 4, (0, 255, 0), -1)
        #  убрать мелкий шум

        if area > self.frame.shape[0]:

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

            # cv2.ellipse(frame, cv2.fitEllipse(cnt), (0, 0, 255), 2)
            # cv2.circle(frame, (int(x1), int(y1)), int(r1), (randrange(255), randrange(255), 255), 2)

            maskTmp = cv2.drawContours(np.zeros(self.img.shape[:2], np.uint8), [cntr], -1, 255, -1)
            mean = cv2.mean(self.img, mask=maskTmp)
            class_p, color_cnt = self.classifier_potatoes([round(x) for x in mean][2:: -1])
            area = round(self.PIXEL_TO_CM ** 2 * cv2.contourArea(cntr))
            rect = cv2.minAreaRect(cntr)
            fraction = self.fraction_classifier(round(min(rect[1][0], rect[1][1])))
            weight = area * 2.5
            weight_fraction = self.weight_classifier(weight)
            self.potatoes.append([str(self.potato_id),
                                  str(area),
                                  str([round(x) for x in mean][2:: -1]),
                                  class_p,
                                  str(self.ellipse_deviation(cntr)),
                                  str(weight),
                                  str(weight_fraction),
                                  str(round(min(rect[1][0], rect[1][1]))),
                                  str(fraction)])

            # cv2.putText(self.frame, ('%02d%02d%02d' % mean[:3])[:self.COLOR_ACCURACY], (int(x1) - 50, int(y1) + 20),
            #             1, 6, (0, 255, 0), 6, cv2.LINE_AA)
            # (randrange(200), 255, randrange(200))
            tmpCof = self.frame.shape[0] / 1300
            # [255 - i for i in [int(x) for x in mean][0:3]]
            cv2.drawContours(self.frame, [cntr], 0, color_cnt, int(7 * tmpCof))
            cv2.putText(self.frame, str(self.potato_id), (int(x1 - 70 * tmpCof), int(y1 + 33 * tmpCof)),
                        3, 3.3 * tmpCof, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(self.frame, str(self.potato_id), (int(x1 - 70 * tmpCof), int(y1 + 33 * tmpCof)),
                        3, 3.3 * tmpCof, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(frame, str([round(i) for i in mean[:3]]), (x, y), 1, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            if self.pointsQR:
                for i in self.pointsQR:
                    sample = cv2.pointPolygonTest(np.array(cntr), i[0], False)
                    if sample == 1:
                        cv2.putText(self.frame, i[1], (int(x1 - 70 * tmpCof), int(y1 + 33 * tmpCof)),
                                    3, 2.3 * tmpCof, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(self.frame, i[1], (int(x1 - 70 * tmpCof), int(y1 + 33 * tmpCof)),
                                    3, 2.3 * tmpCof, (0, 255, 0), 1, cv2.LINE_AA)
                        tmp1 = (4 / 3) * self.PIXEL_TO_CM ** 2 * cv2.contourArea(cntr) * (
                            min(rect[1][0], rect[1][1] * self.PIXEL_TO_CM / 2))
                        if len(i[1]) > 4:
                            [a, b, c] = [int(x) for x in i[1].split(',')]
                            tmp2 = (4 / 3) * np.pi * a * b * c
                            print(str(int(abs(tmp1 - tmp2) / tmp2 * 100)) + '% - Погрешность объема/массы')

                        print(i[1])

            self.potato_id += 1
        else:
            cv2.putText(self.belt, str('x'), (int(x1) - 10, int(y1)), 1, 6, (0, 255, 0), 6, cv2.LINE_AA)

    @staticmethod
    def weight_classifier(weight):
        if weight > 140:
            return 150
        elif weight > 120:
            return 130
        elif weight > 100:
            return 110
        elif weight > 80:
            return 90
        elif weight > 60:
            return 70
        elif weight > 40:
            return 50
        elif weight > 20:
            return 30
        else:
            return 10

    @staticmethod
    def fraction_classifier(min_diameter):
        if min_diameter > 80:
            return 80
        elif min_diameter > 70:
            return 75
        elif min_diameter > 60:
            return 65
        elif min_diameter > 50:
            return 55
        elif min_diameter > 40:
            return 45
        elif min_diameter > 30:
            return 35
        elif min_diameter > 20:
            return 25
        elif min_diameter > 10:
            return 15
        else:
            return 5

    @staticmethod
    def classifier_potatoes(color):
        [R, G, B] = color
        if R > 190:
            return 'Красня', [0, 0, 255]
        elif G > 180:
            return 'Зеленая', [0, 255, 0]
        else:
            return 'Белая', [255, 255, 255]

    @staticmethod
    def ellipse_deviation(cntr):
        """
        Метод вычисляет отклонение формы клубня от эллипсоидной формы
        :param cntr: Контур анализируемого клубня
        :return: Отклонение от эллипса в сантиметрах
        """
        # cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(belt, ellipse, (0, 0, 255), 2)
        hull = cv2.convexHull(cntr, False)
        if len(hull) < 5:
            return -1
        ellipse = cv2.fitEllipse(hull)
        areaCnt = round(cv2.contourArea(cntr))
        areaEllipse = round(np.pi / 4 * ellipse[1][0] * ellipse[1][1])
        # areaHull = int(cv2.contourArea(hull))
        if areaEllipse > 0:
            return round((areaCnt / areaEllipse) * 100)
        else:
            return -1

    def create_report(self):
        """
        Метод сохраняет данные о клубнях в таблицу table.csv
        :return: None
        """
        # # print(self.potatoes)
        # f = open('table.csv', 'w')
        # for i in self.potatoes:
        #     f.write(str(i) + '\n')
        # f.close()
        pass


if __name__ == '__main__':
    # Создание объекта класса анализатора фото
    # 'data/5.jpg', 'data/1.mp4'
    imgAnalyzer = ImageProcessor(srcImg='data/8.jpg', ratio=1)
    # imgAnalyzer = ImageProcessor(camera=0, ratio=1)
    imgAnalyzer.aruco_marker()
    imgAnalyzer.qr_code_detector()
    num = 0
    key = 0

    # Выход при нажалии Esc
    while key != 27:
        # (0, 0, 33, 102, 85, 255, 2, 20) # 5.jpg
        imgAnalyzer.find_and_draw_contours()
        imgAnalyzer.resize(1)
        imgAnalyzer.showAll()

        key = imgAnalyzer.get_key(num)
        num = 1
