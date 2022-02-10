from collections import Counter
from pymongo import MongoClient

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QTableView, QDialog, QPushButton, QVBoxLayout

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
            tmp = i[1:-1].split(',')
            color_mean[0] += int(tmp[0])
            color_mean[1] += int(tmp[1])
            color_mean[2] += int(tmp[2])
        c_len = len(color_list)
        print(color_mean)
        color_mean = [int(color_mean[0] / c_len),
                      int(color_mean[1] / c_len),
                      int(color_mean[2] / c_len)]
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
