import cv2
import numpy as np

icol = (20, 5, 20)


# class ColorBar:

# @staticmethod
def nothing(x):
    pass


cv2.namedWindow("Blur")
cv2.createTrackbar("a", "Blur", icol[0], 50, nothing)  # создание элемента  Trackbar
cv2.createTrackbar("b", "Blur", icol[1], 50, nothing)
cv2.createTrackbar("c", "Blur", icol[2], 50, nothing)


def getABC():
    a = cv2.getTrackbarPos("a", "Blur")
    b = cv2.getTrackbarPos("b", "Blur")
    c = cv2.getTrackbarPos("c", "Blur")

    return a, b, c


if __name__ == '__main__':
    img = cv2.imread('3.jpg')
    img = cv2.resize(img, (711, 400))
    while True:
        frame = img.copy()

        # (w, h, c) = frame.shape
        # frame = cv2.resize(frame, (int(h / 3), int(w / 3)))
        a, b, c = getBlur()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        cv2.imshow("qwe", blurred)

        # detected_edges = cv2.Canny(blurred, 53, 61)
        detected_edges = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        cv2.erode(detected_edges, element)
        cv2.imshow("qwqq", detected_edges)

        # mask = detected_edges != 0
        # dst = img * (mask[:, :, None].astype(img.dtype))
        # cv2.imshow("Frame", detected_edges)

        # g = GaussianMixture(n_components=14)
        # g.fit(X)
        #
        # plt.scatter(X[:, 0], X[:, 1], c=g.predict(X))
        # plt.show()

        # mask = detected_edges != 0
        # dst = img * (mask[:, :, None].astype(img.dtype))


        key = cv2.waitKey(1)
        if key == 27:
            break
