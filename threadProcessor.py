from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from cv2 import cv2

import numpy as np

from colorBar import getColor, getBlur, getWatershedSens


class ThreadProcessor:
    def __init__(self, camera, ratio):
        self.ratio = ratio
        self.cap = cv2.VideoCapture(camera)
        self.update_frame()
        self.frame = self.img.copy()

    def update_frame(self):
        """
        Обновляет кадр (Только для видео)
        :return: None
        """
        _, self.img = self.cap.read()
        (w, h, c) = self.img.shape
        self.img = cv2.resize(self.img, (int(h * self.ratio), int(w * self.ratio)))
        self.frame = self.img.copy()

    def resize(self, num):
        """
        Метод изменения размера окана
        :param num: Коэффициент уменьшения (Множитель)
        :return: None
        """
        (w, h, c) = self.img.shape
        # self.img = cv2.resize(self.img, (int(h / 4), int(w / 4)))
        self.frame = cv2.resize(self.frame, (int(h * num), int(w * num)))

    def showAll(self):
        """
        Метод вывода информационных окон
        :return: None
        """
        cv2.imshow("Frame", self.frame)

    def get_key(self, num):
        """
        Метод для вызова паузы и считывания введенного символа
        :param num: При значении 0 ожидает ввода люього символа
        :return: Символ который был введен
        """
        key = cv2.waitKey(num)
        return key


if __name__ == '__main__':
    threadProcessor = ThreadProcessor(camera=0, ratio=1)
    key = 0
    # Выход при нажалии Esc
    while key != 27:
        threadProcessor.resize(1)
        threadProcessor.showAll()
        threadProcessor.update_frame()

        key = threadProcessor.get_key(1)
