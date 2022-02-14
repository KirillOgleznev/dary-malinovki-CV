from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QAction, QLabel, QMainWindow, QMessageBox
from cv2 import cv2

from src.colorBar import TrackingBar
from src.imageProcessor import ImageProcessor
from src.interfaceQT.dataDialog import ClassDialog
from src.interfaceQT.areaSelectionDialog import AreaSelectionDialog


class Thread(QThread):
    changePixmap = pyqtSignal(QImage, list, ImageProcessor)

    def run(self):
        imgAnalyzer = ImageProcessor(camera='realsense')
        # imgAnalyzer = ImageProcessor(srcImg='data/depth.png')
        while True:
            imgAnalyzer.update_frame()
            # imgAnalyzer.aruco_marker()
            # imgAnalyzer.qr_code_detector()

            imgAnalyzer.find_and_draw_contours()
            # imgAnalyzer.showAll(frame=False)
            cv2.waitKey(1)
            frame = imgAnalyzer.getFrame()
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(w, h, 1)
            self.changePixmap.emit(p, imgAnalyzer.potatoes, imgAnalyzer)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'AnalysisWindow'

        self.potatoesList = [ImageProcessor.COLUMNS_NAMES]
        self.imageProcessor = None
        self.flagFrameUpdate = True
        self.areaSelectionDialogOpen = False
        self.areaSelectionDialog = None
        self.createMenus()
        self.createToolBars()

        self.setWindowTitle(self.title)

        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()

        self.show()

    def createToolBars(self):
        self.fileToolBar = self.addToolBar("Опции")
        analysisAct = QAction("Анализ", self, triggered=self.btnAnalysisClicked, font=QFont('Times', 14))
        self.fileToolBar.addAction(analysisAct)

        area_selection = QAction("Область", self, triggered=self.btnAreaSelectionClicked, font=QFont('Times', 14))
        self.fileToolBar.addAction(area_selection)

        # btn1 = QPushButton("Анализ", self)
        # btn1.move(self.width // 2 - 50, self.height - 60)
        # btn1.clicked.connect(self.btnAnalysisClicked)

    def createMenus(self):
        settingsMenu = self.menuBar().addMenu("&Настройки")
        segmentationAct = QAction("Настройка сегментации", self, triggered=self.segmentator)
        settingsMenu.addAction(segmentationAct)
        calibratorAct = QAction("Калиброва фона", self, triggered=self.calibrator)
        settingsMenu.addAction(calibratorAct)

    def calibrator(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Убедитесь, что в области анализа нет объектов!")
        msgBox.setWindowTitle("Information")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.imageProcessor.calibratorProcess()
        else:
            print('Cancel clicked')

    def segmentator(self):
        pass
        # self.flagFrameUpdate = False
        # TrackingBar(self)

        # dialog.exec_()

    @pyqtSlot(QImage, list, ImageProcessor)
    def setImage(self, image, potatoes, imageProcessor):
        if self.flagFrameUpdate:
            self.imageProcessor = imageProcessor
            self.label.setPixmap(QPixmap.fromImage(image))
            self.potatoesList = potatoes
        if self.areaSelectionDialogOpen:
            self.areaSelectionDialog.setImage()

    def btnAreaSelectionClicked(self):
        self.areaSelectionDialogOpen = True
        self.areaSelectionDialog = AreaSelectionDialog(self, self.imageProcessor)
        self.areaSelectionDialog.exec_()
        self.areaSelectionDialogOpen = False

    def btnAnalysisClicked(self):
        self.flagFrameUpdate = False
        # f = open('table.csv', 'w')
        # for i in self.potatoesList:
        #     f.write(str(i) + '\n')
        # f.close()
        dialog = ClassDialog(self)
        dialog.exec_()
