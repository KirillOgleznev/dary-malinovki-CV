from cv2 import cv2
import sys
from PyQt5.QtWidgets import QTableView, QDialog, QWidget, QLabel, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap

from imageProcessor import ImageProcessor
from collections import Counter
from pymongo import MongoClient

password = 'secretPassword'
connect = MongoClient('mongodb://Admin:' + password + '@140.238.219.37:27017')
test_collection = connect['dm_db']['test']


class ClassDialog(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.title = 'Table'
        self.setWindowTitle("Dialog")
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
        for i in listPotatoes:
            tmp_varieties.append(i[3])
            tmp_weight_fraction.append(i[6])
            tmp_size_fraction.append(i[8])
            if int(i[4]) < 85:
                ugly_shape += 1
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
    changePixmap = pyqtSignal(QImage, list)

    def run(self):
        imgAnalyzer = ImageProcessor(srcImg='data/5.jpg', ratio=0.5)
        while True:
            imgAnalyzer.update_frame()
            imgAnalyzer.aruco_marker()
            imgAnalyzer.find_and_draw_contours()
            imgAnalyzer.showAll(frame=False)
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
            self.changePixmap.emit(p, imgAnalyzer.potatoes)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'AnalysisWindow'
        self.left = 0
        self.top = 0
        self.width = 640
        self.height = 700

        self.potatoesList = [ImageProcessor.COLUMNS_NAMES]
        self.flagFrameUpdate = True

        self.initUI()

    @pyqtSlot(QImage, list)
    def setImage(self, image, potatoes):
        if self.flagFrameUpdate:
            self.label.setPixmap(QPixmap.fromImage(image))
            self.potatoesList = potatoes

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.resize(self.width, self.height - 100)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

        btn1 = QPushButton("Анализ", self)
        btn1.move(self.width // 2 - 50, self.height - 60)
        btn1.clicked.connect(self.button1Clicked)

        self.show()

    def button1Clicked(self):
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
