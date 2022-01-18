from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject
import numpy as np

from cv2 import cv2


# initialize the video writer (we'll instantiate later if need be)
writer = None
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None


ct = CentroidTracker(maxDisappeared=1000)
trackers = []
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
# start the frames per second throughput estimator
cap = cv2.VideoCapture(1)
# loop over frames from the video stream
report = []
while True:
    # read the next frame from the video stream and resize it
    _, frame = cap.read()
    (H, W, _) = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    status = "Waiting"

    RGB_belt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_belt = cv2.cvtColor(RGB_belt, cv2.COLOR_BGR2HSV)
    from colorBar import getColor, getBlur
    lowHue, lowSat, lowVal, highHue, highSat, highVal = getColor()
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    threshold = cv2.inRange(hsv_belt, colorLow, colorHigh)
    threshold = cv2.medianBlur(threshold, getBlur())
    # threshold = 255 - threshold

    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    centrs = []
    cntr = []
    cY = 0
    for cnt in cnts:
        if len(cnt) < 50:
            continue
        # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cntr.append(cnt)
        cX = int((x + x + w) / 2.0)
        cY = int((y + y + h) / 2.0)
        centrs.append(str([cX, cY]))
        rects.append((x, y, x + w, y + h))

    # Закинуть сюда прямоугольники

    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            if abs(centroid[1] - to.centroids[-1][1]) > 100:
                ct.deregister(objectID)
                del trackableObjects[objectID]
                break
            to.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        if str([centroid[0], centroid[1]]) in centrs:
            tmpIndex = centrs.index(str([centroid[0], centroid[1]]))
            tmpCnt = cntr[tmpIndex]
            maskTmp = cv2.drawContours(np.zeros(frame.shape[:2], np.uint8), [tmpCnt], -1, 255, -1)
            cv2.imshow('maskTmp', maskTmp)
            mean = cv2.mean(frame, mask=maskTmp)

            hull = cv2.convexHull(tmpCnt, False)
            if len(hull) > 6:
                ellipse = cv2.fitEllipse(hull)
                areaCnt = int(cv2.contourArea(tmpCnt))
                areaEllipse = int(np.pi / 4 * ellipse[1][0] * ellipse[1][1])
                # areaHull = int(cv2.contourArea(hull))
                report.append([objectID, '%02d%02d%02d' % mean[:3], int((areaEllipse - areaCnt) * 0.002)])
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown)
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1

# stop the timer and display FPS information
f = open('table.csv', 'w')
for i in report:
    f.write(str(i) + '\n')
f.close()
# close any open windows
cv2.destroyAllWindows()
