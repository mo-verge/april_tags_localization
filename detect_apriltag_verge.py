import cv2
import numpy as np
import matplotlib.pyplot as ppl
import pupil_apriltags as apriltag
import math

def PolyArea2D(pts):
    l = np.hstack([pts, np.roll(pts, -1, axis=0)])
    a = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in l))
    return a

dt = 1/30
def low_pass(x_new, y_old, cutoff=0.1):
    alpha = dt / (dt + 1 / (2 * np.pi * cutoff))
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new

yfilt_old = 0
def plotCamera3D(Cesc, rvec, ax=None):
    point = ax.scatter3D(Cesc[0], Cesc[1], Cesc[2], 'k', c='red')
    R, _ = cv2.Rodrigues(rvec)

    p1_cam = [-20, 20, 50]
    p2_cam = [20, 20, 50]
    p3_cam = [20, -20, 50]
    p4_cam = [-20, -20, 50]

    p1_esc = R.T @ p1_cam + Cesc
    p2_esc = R.T @ p2_cam + Cesc
    p3_esc = R.T @ p3_cam + Cesc
    p4_esc = R.T @ p4_cam + Cesc
    camera_plot = [ax.plot3D((Cesc[0], p1_esc[0]), (Cesc[1], p1_esc[1]), (Cesc[2], p1_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p2_esc[0]), (Cesc[1], p2_esc[1]), (Cesc[2], p2_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p3_esc[0]), (Cesc[1], p3_esc[1]), (Cesc[2], p3_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p4_esc[0]), (Cesc[1], p4_esc[1]), (Cesc[2], p4_esc[2]), '-k'),
                   ax.plot3D((p1_esc[0], p2_esc[0]), (p1_esc[1], p2_esc[1]), (p1_esc[2], p2_esc[2]), '-k'),
                   ax.plot3D((p2_esc[0], p3_esc[0]), (p2_esc[1], p3_esc[1]), (p2_esc[2], p3_esc[2]), '-k'),
                   ax.plot3D((p3_esc[0], p4_esc[0]), (p3_esc[1], p4_esc[1]), (p3_esc[2], p4_esc[2]), '-k'),
                   ax.plot3D((p4_esc[0], p1_esc[0]), (p4_esc[1], p1_esc[1]), (p4_esc[2], p1_esc[2]), '-k')]

    return camera_plot, point


def getCamera3D(rvec, tvec):
    # Centro óptico de la cámara como un punto 3D expresado en el sistema de la escena
    # t = -R @ Cesc => Cesc = -R^-1 @ t, pero R^-1 = R.T => Cesc = -R.T @ t
    R, _ = cv2.Rodrigues(rvec)
    Cesc = (-R.T @ tvec).reshape(3)

    return Cesc


npz_file = "calibration.npz"
tagsize = 45.3
family = "tagStandard52h13"
camera = 0
ids = [7, 57]
# objectPoints = {7:np.array([[0., 0., 0.], [tagsize, 0., 0.], [tagsize, tagsize, 0.], [0., tagsize, 0.]]),
                # 57:np.array(
                    # [[150.0, 0., 0.], [150.0 + tagsize, 0., 0.], [150.0 + tagsize, tagsize, 0,],
                     # [150.0, tagsize, 0]])}
objectPoints = {7:np.array([ [0., tagsize, 0.], [tagsize, tagsize, 0.], [tagsize, 0., 0.], [0., 0., 0.]]),
                57:np.array([ [0., tagsize, 0.], [tagsize, tagsize, 0.], [tagsize, 0., 0.], [0., 0., 0.]])}

with np.load(npz_file) as data:
    intrinsics = data['intrinsics']
    dist_coeffs = data['dist_coeffs']

vs = cv2.VideoCapture(camera)
detector = apriltag.Detector(families=family)
fig = ppl.figure(figsize=(3, 3))
axes = ppl.axes(projection='3d')
axes.set_xlabel('X (mm)')
axes.set_ylabel('Y (mm)')
axes.set_zlabel('Z (mm)')

for tagID, objectPoint in objectPoints.items():
    axes.scatter3D(objectPoint[0, 0], objectPoint[0, 1], objectPoint[0, 2], '-k', c='blue')

    axes.plot3D((objectPoint[0, 0], objectPoint[1, 0]), (objectPoint[0, 1], objectPoint[1, 1]),
                (objectPoint[0, 2], objectPoint[1, 2]), '-g')
    axes.plot3D((objectPoint[1, 0], objectPoint[2, 0]), (objectPoint[1, 1], objectPoint[2, 1]),
                (objectPoint[1, 2], objectPoint[2, 2]), '-g')
    axes.plot3D((objectPoint[2, 0], objectPoint[3, 0]), (objectPoint[2, 1], objectPoint[3, 1]),
                (objectPoint[2, 2], objectPoint[3, 2]), '-g')
    axes.plot3D((objectPoint[3, 0], objectPoint[0, 0]), (objectPoint[3, 1], objectPoint[0, 1]),
                (objectPoint[3, 2], objectPoint[0, 2]), '-g')

camera_points = []

# vs.set(cv2.CAP_PROP_FOURCC, -1)
vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while vs.isOpened():
    lines = []
    ret, image = vs.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    coord_fusion = []
    angle_fusion = []
    areas = []
    cameras = {}
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        imagePoints = r.corners

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
        cv2.circle(image, ptA, 5, (255, 0, 0), -1)

        # draw the tag id on the image
        tagid = "tag_id = " + str(r.tag_id)
        cv2.putText(image, tagid, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, rotation, translation = cv2.solvePnP(objectPoints[r.tag_id], imagePoints, intrinsics, dist_coeffs)
        # M = np.empty((4, 4))
        # M[:3, :3] = rotation
        # M[:3, 3] = [translation[0][0], translation[1][0], translation[2][0]]
        # M[3, :] = [0, 0, 0, 1]
        # cameras[r.tag_id] = np.matmul(M, [0,0,0,1])
        cameras[r.tag_id] = [t[0] for t in translation]

    if len(cameras.keys()) == 2:
        x1,y1,z1 = cameras[7]
        x2,y2,z2 = cameras[57]
        yraw = math.sqrt(pow((x2 - x1),2)+ pow((y2-y1),2)+ pow((z2-z1),2))
        yfilt = low_pass(yraw, yfilt_old )
        yfilt_old = yfilt
        print(f"{yfilt:.2f}, {yraw:.2f}")

    cv2.imshow("camera", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
