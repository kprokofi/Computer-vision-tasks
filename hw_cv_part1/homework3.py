import numpy as np
import math
from scipy import linalg
import cv2 as cv

print("__TASK1__")
alpha = 45*math.pi/180
R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
              [np.sin(alpha), np.cos(alpha), 0],
              [0, 0, 1]])
T = np.array([10,0,0]).T
F_1 = np.cross(T, R)
print("F=\n", np.round(F_1, 3))

print("__TASK2__")
alpha1 = 45*math.pi/180
alpha2 = -45*math.pi/180

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

# F = [e2]_x*P2*P1_inversed
# inverse matrix P1
U, E, V_T = np.linalg.svd(P1, full_matrices=False)
P1_inversed = np.matmul(np.matmul(V_T.T, np.diag(1/E)), U.T)
# e2 = P2*O ==> we do not have translation between first camera
# and world coordinate system, then O is supposed to be set to [0,0,0].T
# but as we have P2 [3,4] we need define the fourth coordinate equals to 1
e2 = np.matmul(P2, np.array([0, 0, 0, 1]).T)
print(e2)
# Calculate F by the given formula above
F = np.cross(e2, np.matmul(P2, P1_inversed),axisa=0, axisb=0, axisc=0)
print("F=\n", np.round(F ,3), " F.T*e': ", np.matmul(F.T, e2))

print("__TASK3__")
# we have rotation and translation of the one camera relative to the second one
# Also we have identety K matrixes. So, we may have get the following equation:
# e1.T*l1 = 0 , l1.T*e1 = 0, l1 = F.T*q2 or l1.T = q2.T*F ==> q2.T*F*e1 = 0
# solve F*e1 = 0 by SVD, F = U*E*V.T
# The solution for e1 is column of the matrix V relative to zero singular values
# or it the same as the row of the matrix V.T relative to zero singular values.
U,E,V_T = np.linalg.svd(F_1, full_matrices=False)
null_mask = E < 1e-6
e1 = V_T[null_mask]
# for e2 we can obtain the same equation:
# q1.T*F.T*e2 = 0
# solve F.T*e2 = 0
U,E,V_T = np.linalg.svd(F_1.T, full_matrices=False)
null_mask = E < 1e-6
e2 = V_T[null_mask]

print("e = \n", np.round(e1, 3))
print("e' = \n", np.round(e2, 3))

print("__TASK4__")
# l1 = e1 x q1
l1 = np.cross(e1.reshape(-1), [0, 0, 1])
# l2 = F*[e1]xl1
l2 = np.matmul(F_1, np.cross(e1.reshape(-1), l1))
print(f"l epipolar line:\n{np.round(l1,3)}\nl' epipolar line:\n{np.round(l2,3)}")

print("__TASK5__")
# see explonations in the handwritten paper
print(f"l' = {[0, -1, 0]}")
