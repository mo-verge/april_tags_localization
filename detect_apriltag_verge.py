import sys
import time

import cv2
import numpy as np
import pupil_apriltags as apriltag
import math
from picamera2 import Picamera2
from libcamera import controls

LOCALIZATION_TAG_ID_A = 7
LOCALIZATION_TAG_ID_B = 57
ROBOT_TAG_ID = 6

def PolyArea2D(pts):
    l = np.hstack([pts, np.roll(pts, -1, axis=0)])
    a = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in l))
    return a

def low_pass(x_new, y_old, dt, cutoff=0.1):
    alpha = dt / (dt + 1 / (2 * np.pi * cutoff))
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new

npz_file = "calibration.npz"
tagsize = 15
family = "tagStandard52h13"
camera = 0
tagDimentions = np.array([ [-tagsize, tagsize, 0.], [tagsize, tagsize, 0.], [tagsize, -tagsize, 0.],
                          [-tagsize, -tagsize, 0.]])

with np.load(npz_file) as data:
    intrinsics = data['intrinsics']
    dist_coeffs = data['dist_coeffs']

print ("Starting camera")
picam2 = Picamera2()

config = picam2.create_video_configuration(
    # raw={"format": 'SRGGB10', 'size': picam2.sensor_resolution},
    main={'size': picam2.sensor_resolution},
    # buffer_count = 4
)
picam2.configure(config)

picam2.start()

picam2.set_controls({
    # "AnalogueGain":0.1,
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
    "HdrMode": controls.HdrModeEnum.Off,
    # "AeEnable": False,
    "AwbEnable": False,
    "AeFlickerMode": controls.AeFlickerModeEnum.Off,
    # "ExposureTime" : 500,
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 0.0})

detector = apriltag.Detector(families=family)
print ("April tag detector started")
print(picam2.camera_controls['AnalogueGain'])
print(picam2.capture_metadata()['LensPosition'])
print (picam2.sensor_resolution)
print ("Camera started")
results = []

doShow = False
# doShow = True

yfilter = 0
while True:
    curr = time.time()
    lines = []

    # image = picam2.capture_array("raw")
    # image = image.view(np.uint16)
    # image = np.maximum(image, 64) - 64
    # image = image >> 2
    # image = image.astype(np.uint8)

    # grayImage = cv2.cvtColor(image, cv2.COLOR_BayerRGGB2GRAY)
    # if doShow:
        # image = cv2.cvtColor(image, cv2.COLOR_BayerRGGB2RGB)

    image = picam2.capture_array("main")
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if doShow:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    coord_fusion = []
    angle_fusion = []
    areas = []
    tagLocationsCamera = {}
    tagLocationsImage = {}

    results = detector.detect(grayImage)
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        imagePoints = r.corners

        if doShow:
            ptA, ptB, ptC, ptD = imagePoints
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            areas.append((PolyArea2D(imagePoints)))

            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)

            # draw the left-down (x, y)-coordinates of the AprilTag
            cv2.circle(image, ptD, 5, (255, 0, 0), -1)

            # draw the tag id on the image
            tagid = "tag_id = " + str(r.tag_id)
            cv2.putText(image, tagid, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            tagLocationsImage[r.tag_id] = imagePoints
        _, rotation, translation = cv2.solvePnP(tagDimentions, imagePoints, intrinsics, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        tagLocationsCamera[r.tag_id] = [t[0] for t in translation]

    yraw = 0
    if LOCALIZATION_TAG_ID_A in tagLocationsCamera and LOCALIZATION_TAG_ID_B in tagLocationsCamera:
        xa,ya,za = tagLocationsCamera[LOCALIZATION_TAG_ID_A]
        xb,yb,zb = tagLocationsCamera[LOCALIZATION_TAG_ID_B]
        yraw = math.sqrt(pow((xb-xa),2) + pow((yb-ya),2) + pow((zb-za),2))

    if LOCALIZATION_TAG_ID_A in tagLocationsCamera and ROBOT_TAG_ID in tagLocationsCamera:
        xa,ya,_ = tagLocationsCamera[LOCALIZATION_TAG_ID_A]
        xb,yb,_ = tagLocationsCamera[ROBOT_TAG_ID]

    if doShow:
        if LOCALIZATION_TAG_ID_A in tagLocationsImage and LOCALIZATION_TAG_ID_B in tagLocationsImage:
            _, _, _, ptA = tagLocationsImage[LOCALIZATION_TAG_ID_A]
            _, _, _, ptB = tagLocationsImage[LOCALIZATION_TAG_ID_B]
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            cv2.line(image, ptA, ptB, (255, 0, 0), 2)
        if ROBOT_TAG_ID in tagLocationsCamera:
            ptA, ptB, ptC, ptD = tagLocationsImage[ROBOT_TAG_ID]
            yOffset = 500
            xOffset = 500
            ptA = (int(ptA[0]) + xOffset, int(ptA[1]) + yOffset)
            ptB = (int(ptB[0]) + xOffset, int(ptB[1]) + yOffset)
            ptC = (int(ptC[0]) + xOffset, int(ptC[1]) + yOffset)
            ptD = (int(ptD[0]) + xOffset, int(ptD[1]) + yOffset)
            cv2.line(image, ptA, ptB, (0, 255, 255), 5)
            cv2.line(image, ptB, ptC, (0, 255, 255), 5)
            cv2.line(image, ptC, ptD, (0, 255, 255), 5)
            cv2.line(image, ptD, ptA, (0, 255, 255), 5)

        cv2.imshow("camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    delta = time.time() - curr
    # yfilter = low_pass(yraw, yfilter, delta, cutoff=0.1)
    print(f"{delta:.4f}, {len(results)}, {yraw:.4f}")

cv2.destroyAllWindows()
