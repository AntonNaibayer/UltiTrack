import glob

import cv2
import numpy as np

PATTERN = (9, 6)  # внутренние углы
SQUARE_M = 0.027  # размер клетки в метрах

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_M

obj_pts, img_pts = [], []
for path in glob.glob("calib/*.jpg"):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN)
    if ok:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_pts.append(objp)
        img_pts.append(corners)

h, w = gray.shape
rms, K, dist, _, _ = cv2.calibrateCamera(
    obj_pts,
    img_pts,
    (w, h),
    None,
    None,
    flags=cv2.CALIB_RATIONAL_MODEL,  # k1..k6, лучше для сильного wide
)
print("RMS reprojection error, px:", rms)  # хорошо: < 0.5
np.savez("calib/camera.npz", K=K, dist=dist, size=(w, h))
