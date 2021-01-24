from scipy.linalg import solve
import cv2 as cv
import numpy as np
# pts1: np.array, pts2: np.array
def FindFundamentalMat(pts1: np.array, pts2: np.array):
    equations = []

    for i in range(pts1.shape[1]):
        u1 = pts1[0][i]
        v1 = pts1[1][i]
        u2 = pts2[0][i]
        v2 = pts2[1][i]
        equations.append([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])

    equations = np.array(equations).T
    U,E,V = np.linalg.svd(equations)
    f = U[:, -1]
    print(np.round(np.matmul(f, equations), 3))
    F = f.reshape(3,3)

    U,E,V_T = np.linalg.svd(F, full_matrices=False)
    indx_min_sing_val = np.argmin(E)
    E[indx_min_sing_val] = 0.
    F_sing = np.round(np.matmul(np.matmul(U,np.diag(E)), V_T), 3)
    return F_sing

def findFundamnentalMatrix(points_list):
  A = []
  for points in points_list:
    x, y = points[0]
    u, v = points[1]
    A.append([u*x, u*y, u, x*v, y*v, v, x, y, 1])

  A = np.asarray(A, dtype=np.float)
  A = A.T

  u, s, vt =  np.linalg.svd(A)
  print("Minimal Singular Value: {}\n".format(s[-1]))
  x_arr = u[:, -1]
  h_i = np.reshape(x_arr, (3,3))
  print("Fundamental matrix:\n{}\n".format(h_i))
  test = x_arr.dot(A)
  test[test < 1e-10] = 0
  print("The next values must be ZERO: {}\n\n".format(test))
  return h_i

def GeneratePts(max_coord=30, num_points=8):
    R1 = np.identity(3)
    T1 = np.zeros(3).T.reshape(3,1)
    R2 = np.identity(3)
    T2 = np.array([10,0,0]).T.reshape(3,1)

    P1 = np.append(R1, T1, axis=1)
    P2 = np.append(R2, T2, axis=1)
    Q = np.append(np.array([np.random.rand(3)*10 for j in range(num_points)]), 
                  np.ones(num_points).reshape(num_points,1), axis = 1)

    q1 = np.matmul(P1, Q.T)
    q2 = np.matmul(P2, Q.T)

    return q1/q1[2], q2/q2[2]

def main():
    pts = GeneratePts()
    print(FindFundamentalMat(pts[0], pts[1]))
main()