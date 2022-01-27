from cv2 import cv2
import sys
from PyQt5.QtWidgets import QAction, QTableView, QDialog, QWidget, QLabel, QApplication, QPushButton, QVBoxLayout, \
    QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QFont

from imageProcessor import ImageProcessor
from collections import Counter
from pymongo import MongoClient
from colorBar import icol, TrackingBar

password = 'secretPassword'
connect = MongoClient('mongodb://Admin:' + password + '@140.238.219.37:27017')
test_collection = connect['dm_db']['test']


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
        self.createTable()

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
            tmp =i[1:-1].split(',')
            color_mean[0] += int(tmp[0])
            color_mean[1] += int(tmp[1])
            color_mean[2] += int(tmp[2])
        c_len = len(color_list)
        print(color_mean)
        color_mean = [int(color_mean[0]/c_len),
                      int(color_mean[1]/c_len),
                      int(color_mean[2]/c_len)]
        print(color_mean)

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
        for row in self.parent.potatoesList:
            items = [QStandardItem(field) for field in row]
            self.model.appendRow(items)

    def closeEvent(self, event):
        self.parent.flagFrameUpdate = True


class Thread(QThread):
    changePixmap = pyqtSignal(QImage, list, ImageProcessor)

    def run(self):
        imgAnalyzer = ImageProcessor(camera=1)
        # imgAnalyzer = ImageProcessor(srcImg='data/5.jpg', ratio=0.5)
        while True:
            imgAnalyzer.update_frame()
            imgAnalyzer.aruco_marker()
            imgAnalyzer.qr_code_detector()

            from colorBar import icol
            imgAnalyzer.find_and_draw_contours(icol)
            # imgAnalyzer.showAll(frame=False)
            cv2.waitKey(1)
            frame = imgAnalyzer.getFrame()
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            cof = 640 / w
            w = int(w * cof)
            h = int(h * cof)
            p = convertToQtFormat.scaled(w, h, Qt.KeepAspectRatio)
            self.changePixmap.emit(p, imgAnalyzer.potatoes, imgAnalyzer)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'AnalysisWindow'

        self.potatoesList = [ImageProcessor.COLUMNS_NAMES]
        self.imageProcessor = None
        self.flagFrameUpdate = True
        self.createMenus()
        self.createToolBars()

        self.initUI()

    def createToolBars(self):
        self.fileToolBar = self.addToolBar("Опции")
        analysisAct = QAction("Анализ", self, triggered=self.btnAnalysisClicked, font=QFont('Times', 14))
        self.fileToolBar.addAction(analysisAct)

        # btn1 = QPushButton("Анализ", self)
        # btn1.move(self.width // 2 - 50, self.height - 60)
        # btn1.clicked.connect(self.btnAnalysisClicked)

    def createMenus(self):
        settingsMenu = self.menuBar().addMenu("&Настройки")
        segmentationAct = QAction("Настройка сегментации", self, triggered=self.segmentator)
        settingsMenu.addAction(segmentationAct)

    def segmentator(self):
        self.flagFrameUpdate = False
        dialog = TrackingBar(self)

        dialog.exec_()

    @pyqtSlot(QImage, list, ImageProcessor)
    def setImage(self, image, potatoes , imageProcessor):
        if self.flagFrameUpdate:
            self.imageProcessor = imageProcessor
            self.label.setPixmap(QPixmap.fromImage(image))
            self.potatoesList = potatoes

    def initUI(self):
        self.setWindowTitle(self.title)

        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

        self.show()

    def btnAnalysisClicked(self):
        self.flagFrameUpdate = False
        f = open('table.csv', 'w')
        for i in self.potatoesList:
            f.write(str(i) + '\n')
        f.close()
        dialog = ClassDialog(self)
        dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
