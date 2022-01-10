import cv2
import numpy as np
from blurBar import getBlur

# cap = cv2.VideoCapture('1.mp4')
cap = cv2.VideoCapture(0)

# Read the video
while (cap.isOpened()):
    ret, frame = cap.read()
    (w, h, c) = frame.shape
    print((w, h, c))
    frame = cv2.resize(frame, (int(h/2), int(w/2)))
    if ret == True:
        a, b, c = getBlur()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Frame', gray)

        detected_edges = cv2.Canny(gray, a, b, c)
        mask = detected_edges != 0
        dst = frame * (mask[:, :, None].astype(frame.dtype))
        cv2.imshow('window_name', dst)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

cv2.waitKey(0)
# Closes all the frames
cv2.destroyAllWindows()
