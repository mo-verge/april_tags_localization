import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
# from pprintpp import pprint as pp

picam2 = Picamera2()


# config = picam2.create_still_configuration()
# config["size"] = picam2.sensor_resolution
# config["raw"]["size"] = picam2.sensor_resolution
config = picam2.create_video_configuration(
    main={'size': picam2.sensor_resolution},
)
picam2.configure(config)
picam2.start()

picam2.set_controls({
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
    "HdrMode": controls.HdrModeEnum.Off,
    "AwbEnable": False,
    "AeFlickerMode": controls.AeFlickerModeEnum.Off,
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 0.0})
time.sleep(5)

if os.system("ls calibs"):
    os.system("mkdir calibs")

while True:
    image = picam2.capture_array("main")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    calibFileName = f"calibs/calib_image_{dt_string}.bmp"
    cv2.imwrite(calibFileName, gray)
    os.system(f"gpicview {calibFileName}")
    time.sleep(0.1)


# [{'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (1536, 864), 'fps': 120.13, 'crop_limits': (768, 432, 3072, 1728), 'exposure_limits': (9, 77208384, None)}, {'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (2304, 1296), 'fps': 56.03, 'crop_limits': (0, 0, 4608, 2592), 'exposure_limits': (13, 112015443, None)}, {'format': SRGGB10_CSI2P, 'unpacked': 'SRGGB10', 'bit_depth': 10, 'size': (4608, 2592), 'fps': 14.35, 'crop_limits': (0, 0, 4608, 2592), 'exposure_limits': (26, 220417486, None)}]
