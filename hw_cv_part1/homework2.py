import numpy as np
import math
import cv2 as cv

def projection_matrix(alpha, inner_params, t):
    # We need to get R and T matrix
    # R matrix is rotation over OZ
    fx,fy,cx,cy = inner_params
    R = np.array([[np.cos(alpha), -np.sin(alpha), 0],[np.sin(alpha),  np.cos(alpha), 0],[0, 0, 1]])
    T = np.array([[0, 0, t]]).T
    K = np.array([[ fx, 0, cx ],
                  [ 0, fy, cy ],
                  [ 0, 0, 1 ]])
    P = np.dot(K, np.append(R, T, axis=1))
    return P

def direct_linear_transform(points_set):
    # x_hat = H * x 
    # x = U*E*V.T --> x_hat = H*U*E*V.T
    # H ~ x_hat * (U*E*V.T)^-1 = x_hat*V*E^-1*U.T since U and V orthogonal matrix
    # points_set = { |x10||x'10|, |x20||x'20|, |x30||x'30| ... }
    #                |x11||x'11|  |x21||x'21|  |x31||x'31|

    X = np.array([[x[0,0], x[0,1], 1] for x in points_set]).T
    X_hat = np.array([[x[1,0], x[1,1], 1] for x in points_set]).T
    U,E,V = np.linalg.svd(X)
    inversed_E = np.diag(1 / E)

    H = ((X_hat.dot(V.T)).dot(inversed_E)).dot(U.T)
    return H

def homography_rotation(alpha):
    # q1 = (R + T * nt / d)* q2 
    # rotation over OX and T = 0 then H = R
    return np.array([[1, 0, 0],
                  [0, np.cos(alpha), -np.sin(alpha)],
                  [0, np.sin(alpha), np.cos(alpha)]])

print("____TASK1____")  
x = np.array([10, -10, 100]).T
P = projection_matrix(math.pi/4, [400, 400, 960, 540], 10)
x_hat = np.dot(P, np.concatenate([x, np.ones(1)]))
pixel = np.round([[p/x_hat[2]] for p in x_hat[:-1]])
print(f"pixel: {pixel[0][0]}, {pixel[1][0]}\n")

print("____TASK3____")
H = homography_rotation(math.pi/6)
print("H: ", np.round(H,3))
