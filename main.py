import csv

from cv2 import cv2
import sys
from PyQt5.QtWidgets import QTableView, QDialog, QWidget, QLabel, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap

from imageProcessor import ImageProcessor


class ClassDialog(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.title = 'Table'
        self.setWindowTitle("Dialog")
        self.setGeometry(100, 100, 600, 570)

        self.verticalLayout = QVBoxLayout(self)
        self.pushButton = QPushButton(self)
        self.pushButton.clicked.connect(self.btnClosed)
        self.pushButton.setText("Close Dialog")
        self.model = QStandardItemModel(self)
        self.tableView = QTableView(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout.addWidget(self.tableView)
        self.verticalLayout.addWidget(self.pushButton)
        self.parent = parent
        self.loadCsv('table.csv')

    def btnClosed(self):
        self.close()

    def loadCsv(self, fileName):
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput, delimiter=';'):
                items = [QStandardItem(field)for field in row]
                self.model.appendRow(items)

    def closeEvent(self, event):
        self.parent.flagFrameUpdate = True


class Thread(QThread):
    changePixmap = pyqtSignal(QImage, list)

    def run(self):
        imgAnalyzer = ImageProcessor(camera=1, ratio=1)
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
