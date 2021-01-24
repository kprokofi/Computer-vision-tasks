import numpy as np
import math
from scipy import linalg
import cv2 as cv
from PIL import Image
import PIL.ExifTags as exiftag
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

print("__TASK1__")
img = Image.open('GOPR01170000.jpg')
exif_data = {exiftag.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in exiftag.TAGS}
сx = exif_data['ExifImageWidth'] / 2
cy = exif_data['ExifImageHeight'] / 2
f = 24.
fx = f / (1 / ((exif_data['XResolution'][0]/exif_data['XResolution'][1])/25.4))
fy = f / (1 / ((exif_data['YResolution'][0]/exif_data['YResolution'][1])/25.4))
K = np.round(np.array([[fx, 0, сx],
                       [0, fy, cy],
                       [0, 0, 1]]), 3)
print("K= \n", K)

print("__TASK2__")
image = cv.imread('GOPR01170000.jpg')
store = cv.FileStorage()
store.open("camera.xml", cv.FileStorage_READ)
mtx = store.getNode('camera_matrix').mat()
dist = store.getNode('distortion_coefficients').mat()
h, w = image.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv.undistort(image, mtx, dist, None, newcameramtx)
cv.imwrite('undistort.jpg', dst)

print("__TASK3__")
alpha1 = 45*np.pi/180
alpha2 = -45*np.pi/180
# matrixes of the translation
T1 = np.array([0,0,0]).reshape(3,1)
T2 = np.array([10,0,0]).reshape(3,1)
# rotation matrix R1
R1 = np.array([[np.cos(alpha1), -np.sin(alpha1), 0],
              [np.sin(alpha1), np.cos(alpha1), 0],
              [0, 0, 1]])
# rotation matrix R2 over the OY
R2 = np.array([[np.cos(alpha2), 0, np.sin(alpha2)],
               [0, 1, 0],
              [-np.sin(alpha2), 0, np.cos(alpha2)]])
# projection matrix P1 and P2
P1 = np.concatenate([R1.T, -np.matmul(R1.T,T1)], axis=1)
P2 = np.concatenate([R2.T, -np.matmul(R2.T,T2)], axis=1)

r1 = R.from_rotvec(np.pi/4 * np.array([0, 0, 1])).as_quat()
r2 = R.from_rotvec(np.pi/4 * np.array([0, 1, 0])).as_quat()


r1 = Quaternion(matrix=R1)
r2 = Quaternion(matrix=R2)
r_delta = r2*r1.inverse

def compute_quatr(t):
    T = T1*(1-t) + T2*t
    return (1-t)*r1 + t*r2, T
t = 0.5
R_new, T_new = compute_quatr(t)
R_new = R_new.rotation_matrix
P_new = np.concatenate([R_new, T_new], axis=1)
print(f'P with t = {t}:\n', P_new)