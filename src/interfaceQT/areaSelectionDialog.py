from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QLabel


class AreaSelectionDialog(QDialog):
    def __init__(self, parent, imageProcessor):
        super().__init__()
        self.setWindowTitle("Выделение области")

        self.imageProcessor = imageProcessor
        self.setGeometry(100, 100, 200, 200)

        self.verticalLayout = QVBoxLayout(self)
        self.pushButton = QPushButton(self)
        self.pushButton.clicked.connect(self.btnClosed)
        self.pushButton.setText("Выбрать область")

        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.label.setFixedSize(100, 100)

        self.red = QLabel(self.label)
        # self.red.setStyleSheet("background-color: rgba(255, 0, 0, 0.4)")
        self.red.setFixedSize(200, 200)
        self.begin = QPoint(0, 0)
        self.end = QPoint(0, 0)
        self.red.paintEvent = self.myPaintEvent
        self.flagRed = True
        self.flagPress = False
        self.red.mousePressEvent = self.pressEvent
        self.red.mouseMoveEvent = self.moveEvent
        self.red.mouseReleaseEvent = self.releaseEvent

        self.verticalLayout.addWidget(self.label)
        # self.verticalLayout.addWidget(self.red)
        self.verticalLayout.addWidget(self.pushButton)
        self.parent = parent

    def btnClosed(self):

        # todo Исправить баг - если точки начала и коца инвертированы
        self.imageProcessor.slasher = [[self.begin.x(), self.begin.y()],
                                              [self.end.x(), self.end.y()]]
        self.reject()

    def setImage(self):

        image = self.parent.imageProcessor.img_
        h, w, ch = image.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        image = convertToQtFormat.scaled(w, h, 1)

        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

        self.label.setFixedSize(image.size())
        self.red.setFixedSize(image.size())
        if self.flagRed:
            self.begin = QPoint(0, 0)
            self.end = QPoint(image.width(), image.height())
            self.flagRed = False

    def myPaintEvent(self, event):
        qp = QPainter(self.red)
        pen = QPen(QColor(255, 0, 0), 4)
        qp.setPen(pen)
        qp.drawRect(QRect(self.begin, self.end))

    def pressEvent(self, event):
        self.flagPress = True
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def moveEvent(self, event):
        if self.flagPress:
            self.end = event.pos()
            self.update()

    def releaseEvent(self, event):
        self.flagPress = False

    def closeEvent(self, event):
        self.reject()
