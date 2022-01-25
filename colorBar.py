import sys
import time

from cv2 import cv2
import numpy as np

from PyQt5.QtWidgets import QDialog, QLabel, QApplication, QPushButton, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

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
# icol = (0, 0, 33, 102, 85, 255, 2, 20)  # 5.jpg


# icol = (56, 60, 80, 103, 255, 255, 0, 20)  # camera
# icol = (4, 54, 155, 22, 164, 207, 2, 37)  # camera 1
icol = (0, 64, 0, 49, 255, 255, 2, 30)  # camera 1 inv


class Thread(QThread):
    changePixmap = pyqtSignal(QImage, QImage, list)
    finished = pyqtSignal()

    def __init__(self, parent):
        # self.parent = parent
        super().__init__(parent)
        self.flag = False
        # self.imageProcessor = imageProcessor

    @staticmethod
    def nothing(x):
        pass

    def run(self):
        myIcol = icol
        cv2.namedWindow("Tracking")
        cv2.resizeWindow('Tracking', 500, 400)
        cv2.createTrackbar("l_h", "Tracking", myIcol[0], 255, self.nothing)
        cv2.createTrackbar("l_s", "Tracking", myIcol[1], 255, self.nothing)
        cv2.createTrackbar("l_v", "Tracking", myIcol[2], 255, self.nothing)
        cv2.createTrackbar("u_h", "Tracking", myIcol[3], 255, self.nothing)
        cv2.createTrackbar("u_s", "Tracking", myIcol[4], 255, self.nothing)
        cv2.createTrackbar("u_v", "Tracking", myIcol[5], 255, self.nothing)
        cv2.createTrackbar("blur", "Tracking", myIcol[6], 10, self.nothing)
        cv2.createTrackbar("ws_sens", "Tracking", myIcol[7], 300, self.nothing)
        cap = cv2.VideoCapture(1)
        while not self.flag:
            cv2.waitKey(1)
            lowHue = cv2.getTrackbarPos("l_h", "Tracking")
            lowSat = cv2.getTrackbarPos("l_s", "Tracking")
            lowVal = cv2.getTrackbarPos("l_v", "Tracking")
            highHue = cv2.getTrackbarPos("u_h", "Tracking")
            highSat = cv2.getTrackbarPos("u_s", "Tracking")
            highVal = cv2.getTrackbarPos("u_v", "Tracking")
            sens = cv2.getTrackbarPos("ws_sens", "Tracking")
            blur = cv2.getTrackbarPos("blur", "Tracking")
            blur = ((blur // 2) * 2) + 1
            # self.changePixmap.emit([lowHue, lowSat, lowVal, highHue, highSat,
            #                         highVal, sens, blur])

            if not self.parent:
                img = cv2.imread('data/5.jpg')
            else:
                # _, img = self.imageProcessor.cap.read()
                _, img = cap.read()

                pass
            # (w, h, c) = img.shape
            # img = cv2.resize(img, (int(h * 0.3), int(w * 0.3)))

            frame = img.copy()

            RGB_belt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
            colorLow = np.array([lowHue, lowSat, lowVal])
            colorHigh = np.array([highHue, highSat, highVal])
            threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
            threshold = cv2.medianBlur(threshold, blur)

            rgbImage1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbImage2 = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage1.shape
            bytesPerLine = ch * w
            convertToQtFormat1 = QImage(rgbImage1.data, w, h, bytesPerLine, QImage.Format_RGB888)
            convertToQtFormat2 = QImage(rgbImage2.data, w, h, bytesPerLine, QImage.Format_RGB888)
            cof = 640 / w
            w = int(w * cof)
            h = int(h * cof)
            img1 = convertToQtFormat1.scaled(w, h, Qt.KeepAspectRatio)
            img2 = convertToQtFormat2.scaled(w, h, Qt.KeepAspectRatio)
            self.changePixmap.emit(img1, img2, [lowHue, lowSat, lowVal, highHue, highSat,
                                                highVal, blur, sens])
        self.finished.emit()


class TrackingBar(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Calibrator")
        self.lowHue = icol[0]
        self.lowSat = icol[1]
        self.lowVal = icol[2]
        self.highHue = icol[3]
        self.highSat = icol[4]
        self.highVal = icol[5]
        self.blur = icol[6]
        self.sens = icol[7]

        self.label1 = QLabel(self)
        self.label2 = QLabel(self)

        self.gbox = QGridLayout()
        self.gbox.addWidget(self.label1, 0, 0)
        self.gbox.addWidget(self.label2, 0, 1)

        self.button1 = QPushButton("Отмена")
        self.button2 = QPushButton("Сохранить")
        self.button1.clicked.connect(self.btnCancel)
        self.button2.clicked.connect(self.btnSave)

        self.gbox.addWidget(self.button1, 1, 0)
        self.gbox.addWidget(self.button2, 1, 1)

        self.setLayout(self.gbox)

        # self.th = Thread(self, self.parent.imageProcessor)
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.finished.connect(self.final)
        self.th.start()

        self.setGeometry(0, 0, 0, 0)
        self.show()

    def btnCancel(self):
        self.th.flag = True

    def btnSave(self):
        global icol
        icol = (int(self.lowHue), int(self.lowSat), int(self.lowVal),
                int(self.highHue), int(self.highSat), int(self.highVal),
                int(self.blur), int(self.sens))
        self.th.flag = True

    @pyqtSlot()
    def final(self):
        self.close()

    @pyqtSlot(QImage, QImage, list)
    def setImage(self, img1, img2, newIcol):
        self.label1.setPixmap(QPixmap.fromImage(img1))
        self.label2.setPixmap(QPixmap.fromImage(img2))
        self.lowHue = newIcol[0]
        self.lowSat = newIcol[1]
        self.lowVal = newIcol[2]
        self.highHue = newIcol[3]
        self.highSat = newIcol[4]
        self.highVal = newIcol[5]
        self.blur = newIcol[6]
        self.sens = newIcol[7]

    def closeEvent(self, event):
        self.parent.flagFrameUpdate = True


if __name__ == '__main__':
    # cap = cv2.VideoCapture(1)
    app = QApplication(sys.argv)
    ex = TrackingBar(None)
    sys.exit(app.exec_())
    # while True:
    #
    #     # _, img = cap.read()
    #     img = cv2.imread('data/5.jpg')
    #     (w, h, c) = img.shape
    #     img = cv2.resize(img, (int(h * 0.3), int(w * 0.3)))
    #
    #     frame = img.copy()
    #
    #     RGB_belt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
    #     lowHue, lowSat, lowVal, highHue, highSat, highVal = trackingBar.getColor()
    #     colorLow = np.array([lowHue, lowSat, lowVal])
    #     colorHigh = np.array([highHue, highSat, highVal])
    #     threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
    #     threshold = cv2.medianBlur(threshold, trackingBar.getBlur())
    #     # threshold = 255 - threshold
    #
    #     cv2.imshow('Frame', frame)
    #     cv2.imshow('threshold', threshold)
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
