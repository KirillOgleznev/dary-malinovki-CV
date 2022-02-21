import math
from collections import Counter

import numpy as np
from pymongo import MongoClient
import pyrealsense2 as rs

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QTableView, QDialog, QPushButton, QVBoxLayout
from pyntcloud import PyntCloud
from cv2 import cv2

password = 'secretPassword'
connect = MongoClient('mongodb://Admin:' + password + '@140.238.219.37:27017')
test_collection = connect['dm_db']['test']
CAMERA = 0


class ClassDialog(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Данные")
        self.setGeometry(100, 100, 1200, 570)

        self.verticalLayout = QVBoxLayout(self)
        self.pushButton = QPushButton(self)
        self.pushButton.clicked.connect(self.btnClosed)
        self.pushButton.setText("Отправить в БД")
        self.model = QStandardItemModel(self)
        self.tableView = QTableView(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout.addWidget(self.tableView)
        self.verticalLayout.addWidget(self.pushButton)
        self.parent = parent
        self.dataList = []

        self.calculateValues()
        self.createTable()

    def calculateValues(self):
        COLUMNS_NAMES = ['id', 'Объем, мм^3', 'RGB', 'Сорт', 'Элипсоидность',
                         'Вес, г', 'Весовая фракционность', 'диагональ, мм (а)',
                         'Размерная фракционность', 'Высота, мм (b)']
        PIXEL_TO_MM = 1.1

        tmp = self.parent.potatoesList
        potatoes = tmp[0]
        aligned_depth_frame = tmp[1]
        slasher = tmp[2]
        aligned_depth_background = tmp[3]
        self.dataList = [COLUMNS_NAMES]
        for i in potatoes:
            idP = i[0]
            cntr = i[1]
            maskP = i[2]
            mean = i[3]
            rect = cv2.minAreaRect(cntr)

            cY = int(rect[0][1])
            selceted_points = [[px, cY] for [px, py] in cntr if
                               cv2.pointPolygonTest(np.array(cntr), (int(px), int(cY)),
                                                    measureDist=False) == 0]
            (ix, iy) = min(selceted_points, key=lambda x: x[0])
            (x, y) = max(selceted_points, key=lambda x: x[0])
            # udist = aligned_depth_frame.get_distance(ix, iy)
            # vdist = aligned_depth_frame.get_distance(x, y)
            udist = aligned_depth_frame.get_distance(int(rect[0][0]), int(rect[0][1]))
            vdist = udist
            a = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            point1 = rs.rs2_deproject_pixel_to_point(a, [ix, iy], udist)
            point2 = rs.rs2_deproject_pixel_to_point(a, [x, y], vdist)

            distMeters = math.sqrt(
                math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
                    point1[2] - point2[2], 2))

            minX = 100
            coord = False
            [[start_x, start_y], [end_x, end_y]] = slasher
            if aligned_depth_background:
                for i in range(len(maskP)):
                    for j in range(len(maskP[i])):
                        if maskP[i][j] != 0:
                            # cv2.circle(self.frame, (j, i), 2, (0, 255, 0), -1)
                            tmp = aligned_depth_frame.get_distance(start_x + j, start_y + i)
                            if minX > tmp != 0:
                                minX = tmp
                                # 362 193
                                coord = (start_x + j, start_y + i)
            a = aligned_depth_frame.get_distance(int(coord[0]), int(coord[1]))
            b = aligned_depth_background.get_distance(int(coord[0]), int(coord[1]))
            distHeight = round(((b - a) * 1000), 2)
            distPixels = math.sqrt((x - ix) ** 2 + (y - iy) ** 2)
            PIXEL_TO_MM = ((distMeters * 1000) / distPixels)
            # print(PIXEL_TO_MM / minX)
            PIXEL_TO_MM = 1.45
            PIXEL_TO_MM = PIXEL_TO_MM * minX
            class_p, color_cnt = self.classifier_potatoes([round(x) for x in mean][2:: -1])

            area = round(PIXEL_TO_MM * PIXEL_TO_MM * cv2.contourArea(cntr))


            fraction = self.fraction_classifier(round(min(rect[1][0], rect[1][1])))

            volumeTubes = (4 / 3) * area * (distHeight / 2)

            weight = volumeTubes * 0.001604
            weight_fraction = self.weight_classifier(weight)

            self.dataList.append([str(idP),
                                  str(round(volumeTubes)),
                                  str([round(x) for x in mean][2:: -1]),
                                  class_p,
                                  str(self.ellipse_deviation(cntr)),
                                  str(round(weight, 2)),
                                  str(weight_fraction),
                                  str(round(min(rect[1][0], rect[1][1]))),
                                  str(fraction),
                                  str(distHeight)])

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
        areaEllipse = round(3.1415 / 4 * ellipse[1][0] * ellipse[1][1])
        # areaHull = int(cv2.contourArea(hull))
        if areaEllipse > 0:
            return round((areaCnt / areaEllipse) * 100)
        else:
            return -1

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
    def classifier_potatoes(color):
        [R, G, B] = color
        if R > 190:
            return 'Красня', [0, 0, 255]
        elif G > 180:
            return 'Зеленая', [0, 255, 0]
        else:
            return 'Белая', [255, 255, 255]

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

    def btnClosed(self):
        listPotatoes = self.parent.potatoesList[1:]
        count = len(listPotatoes)
        ugly_shape = 0
        tmp_varieties = []
        tmp_weight_fraction = []
        tmp_size_fraction = []
        color_list = []
        for i in listPotatoes:
            tmp_varieties.append(i[3])
            tmp_weight_fraction.append(i[6])
            tmp_size_fraction.append(i[8])
            color_list.append(i[2])
            if int(i[4]) < 85:
                ugly_shape += 1

        color_mean = [0, 0, 0]
        for i in color_list:
            tmp = i[1:-1].split(',')
            color_mean[0] += int(tmp[0])
            color_mean[1] += int(tmp[1])
            color_mean[2] += int(tmp[2])
        c_len = len(color_list)

        color_mean = [int(color_mean[0] / c_len),
                      int(color_mean[1] / c_len),
                      int(color_mean[2] / c_len)]

        varieties = dict(Counter(tmp_varieties))
        weight_fraction = dict(Counter(tmp_weight_fraction))
        size_fraction = dict(Counter(tmp_size_fraction))
        new_record = {
            "count": count,
            "varieties": varieties,
            "weight_fraction": weight_fraction,
            "size_fraction": size_fraction,
        }
        try:
            test_collection.insert_one(new_record)
        except Exception as err:
            print('Не удалось подключиться к БД (' + str(err.details['errmsg']) + ')')

    def createTable(self):
        for row in self.dataList:
            items = [QStandardItem(field) for field in row]
            self.model.appendRow(items)

    def closeEvent(self, event):
        self.parent.flagFrameUpdate = True
