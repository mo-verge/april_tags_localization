import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

picam2 = Picamera2()


config = picam2.create_still_configuration()
config["size"] = picam2.sensor_resolution
config["raw"]["size"] = picam2.sensor_resolution
picam2.configure(config)
picam2.start()

picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 2.0})
time.sleep(1)
# # picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
print(picam2.camera_controls['LensPosition'])
print(picam2.capture_metadata()['LensPosition'])
print (picam2.sensor_resolution)

if os.system("ls calibs"):
    os.system("mkdir calibs")

rgb = picam2.capture_array("main")
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Camera", rgb)
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
calibFileName = f"calibs/calib_image_{dt_string}.bmp"
cv2.imwrite(calibFileName, gray)
os.system(f"gpicview {calibFileName}")


