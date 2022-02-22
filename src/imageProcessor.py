from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from cv2 import cv2
import pyrealsense2 as rs
import numpy as np
import pickle

BLUR_CONST = 19
LOCAL_MAX_CONST = 7


def draw_circle(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDOWN):
        # dist = aligned_depth_frame.get_distance(x, y)
        #
        # dist = round(dist * 100, 2)
        print(x, y)


class ImageProcessor:
    COLOR_ACCURACY = -5
    # Множитель преобразования пикселей в сантиметры
    PIXEL_TO_CM = 0.1029
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters_create()

    def __init__(self, srcImg=None, srcVideo=None, camera=0, ratio=1.0, img=None, slasher=None):
        """

        :param srcImg: Путь до изображения
        :param srcVideo: Путь до видео
        :param camera: Номер камеры подключенной к ПК
        :param ratio: Множитель изменения входящего изображения (можно использовать для оптимизации)
        """
        self.camera = camera
        self.slasher = slasher
        self.pointsQR = []
        self.ratio = ratio
        self.depth_colormap = None
        self.depth_image = None
        self.aligned_depth_frame = None

        self.depth_background = None
        if camera == 'realsense':
            self.pipeline = rs.pipeline()
            config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)

            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            clipping_distance_in_meters = 1  # 1 meter
            self.clipping_distance = clipping_distance_in_meters / self.depth_scale
            self.clipping_distance = 665
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.visual_preset, 0)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
            self.colorizer.set_option(rs.option.min_distance, 0.6)
            self.colorizer.set_option(rs.option.max_distance, 0.7)
            self.colorizer.set_option(rs.option.color_scheme, 2)
            self.update_frame()

        elif img:
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
        self.img_ = self.img.copy()
        self.potato_id = 0
        self.potatoes = []
        self.hsv_belt = None
        self.threshold = None
        self.belt = self.frame.copy()
        # self.shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    def getPotatoes(self):
        return [self.potatoes, self.aligned_depth_frame, self.slasher,
                self.depth_background, self.depth_scale, self.depth_colormap,
                self.depth_image]

    def getFrame(self):
        return self.frame

    def qr_code_detector(self):
        self.pointsQR = []
        det = cv2.QRCodeDetector()
        # ret, cnt = det.detectMulti(np.hstack([self.img, self.img]))
        decodedText, points, _ = det.detectAndDecode(self.img)
        # print(len(decodedText), decodedText)
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
        if self.camera == 'realsense':
            frames = self.pipeline.wait_for_frames()
            frames.keep()
            aligned_frames = self.align.process(frames)
            aligned_frames.keep()

            self.aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            self.depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_image_3d = np.dstack((self.depth_image,
                                        self.depth_image,
                                        self.depth_image))
            # bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color,
            # color_image)

            depth_colormap = np.asanyarray(self.colorizer.colorize(self.aligned_depth_frame).get_data())

            if self.depth_background is None:
                with open('dump.txt', 'rb') as f:
                    tmp = pickle.load(f)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tmp, alpha=0.03), cv2.COLORMAP_JET)
                self.depth_background = np.dstack((tmp,
                                                   tmp,
                                                   tmp))
                depth_colormap_removed = np.where((depth_image_3d > self.depth_background - 10) | (depth_image_3d <= 0),
                                                  0, depth_colormap)

                # depth_colormap_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), 0,
                #                                   depth_colormap)
            else:
                depth_colormap_removed = np.where((depth_image_3d > self.depth_background - 10) | (depth_image_3d <= 0),
                                                  0, depth_colormap)
                # depth_colormap_removed = np.where((depth_image_3d > self.depth_background - 10) | (depth_image_3d <= 0),
                #                                   0, depth_colormap)
            self.depth_colormap = depth_colormap_removed
            self.img = color_image
            self.img_ = color_image

        elif self.cap:
            rav, self.img = self.cap.read()
            if rav:
                (h, w, c) = self.img.shape
                self.img = cv2.resize(self.img, (int(w * self.ratio), int(h * self.ratio)))
        if not self.slasher:
            (h, w, c) = self.img_.shape
            self.slasher = [[67, 0], [w, h - 10]]
            [[start_x, start_y], [end_x, end_y]] = [[67, 0], [w, h - 10]]
            self.img = self.img_[start_y:end_y, start_x:end_x]
        (h, w, c) = self.img_.shape
        cv2.circle(self.img, (int(w / 2), int(h / 2)), 4, (0, 0, 255), -1)

        [[start_x, start_y], [end_x, end_y]] = self.slasher
        self.img = self.img_[start_y:end_y, start_x:end_x]
        self.depth_image = self.depth_image[start_y:end_y, start_x:end_x]
        if self.depth_colormap.any() is not None:
            self.depth_colormap = self.depth_colormap[start_y:end_y, start_x:end_x]


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
        if data is None:
            gray = np.mean(self.depth_colormap, -1)
            mask = gray > gray.mean()
            label_im, nb_labels = ndimage.label(mask)
            gray3 = cv2.blur(gray, (BLUR_CONST, BLUR_CONST))
            local_maxi = peak_local_max(gray3, min_distance=LOCAL_MAX_CONST, labels=label_im)
            mask = np.zeros(gray.shape, dtype=bool)
            mask[tuple(local_maxi.T)] = True
            markers, _ = ndimage.label(mask)
            return watershed(-gray, markers, mask=label_im)
        else:
            RGB_belt = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
            icol = data
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
        self.potatoes = []
        self.frame = self.img.copy()
        self.belt = self.img.copy()

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
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        # todo определить границу шума
        if area > 300:

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

            # cv2.drawContours(frame, [hull], 0, (randrange(255), randrange(255), randrange(255)), 2)

            # cv2.ellipse(frame, cv2.fitEllipse(cnt), (0, 0, 255), 2)
            # cv2.circle(frame, (int(x1), int(y1)), int(r1), (randrange(255), randrange(255), 255), 2)

            maskTmp = cv2.drawContours(np.zeros(self.img.shape[:2], np.uint8), [cntr], -1, 255, -1)
            mean = cv2.mean(self.img, mask=maskTmp)
            class_p, color_cnt = self.classifier_potatoes([round(x) for x in mean][2:: -1])
            rect = cv2.minAreaRect(cntr)

            cY = int(rect[0][1])
            selceted_points = [[px, cY] for [px, py] in cntr if
                               cv2.pointPolygonTest(np.array(cntr), (int(px), int(cY)),
                                                    measureDist=False) == 0]

            if selceted_points:
                left_point = min(selceted_points, key=lambda x: x[0])
                right_point = max(selceted_points, key=lambda x: x[0])
                # cv2.line(self.frame, tuple(left_point), tuple(right_point), [0, 255, 0], 2)

                self.potatoes.append([self.potato_id, cntr, maskTmp, mean])
            else:
                print('err')

            # cv2.putText(self.frame, ('%02d%02d%02d' % mean[:3])[:self.COLOR_ACCURACY], (int(x1) - 50, int(y1) + 20),
            #             1, 6, (0, 255, 0), 6, cv2.LINE_AA)
            # (randrange(200), 255, randrange(200))
            tmpCof = self.frame.shape[0] / 1300
            # [255 - i for i in [int(x) for x in mean][0:3]]
            cv2.drawContours(self.frame, [cntr], 0, color_cnt, int(7 * tmpCof))

            # self.potato_id = round((cv2.contourArea(cntr))/10)
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
                        rect = cv2.minAreaRect(cntr)
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

    def calibratorProcess(self):

        tmp = self.depth_image
        depth_background = np.dstack((tmp, tmp, tmp))
        self.depth_background = depth_background

        # ply = rs.save_to_ply("test1.ply")
        # ply.set_option(rs.save_to_ply.option_ply_binary, True)
        # ply.set_option(rs.save_to_ply.option_ply_normals, False)
        # ply.process(self.aligned_depth_background)
        # with open('dump.txt', 'wb') as f:
        #     f.write(tmp)
        with open('dump.txt', 'wb') as f:
            # Step 3
            pickle.dump(tmp, f)


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
