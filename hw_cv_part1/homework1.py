import numpy as np
import numpy.linalg as npl

def closest_matrix(A: np.array):
    U,E,V = npl.svd(A)
    return np.dot(U, V)

def create_elem(A, i, j, task='second'):
    if task == 'second':
        A[i-1][j-1] = 1/(i + j - 1)
    else:
        A[i-1][j-1] = i + j - 1

def inverse_matrix(A: np.array):
     U,E,V = npl.svd(A)
     # by the properties of the diag matrix
     E_inversed = np.diag(E**(-1))
     # pseudo inversed matrix for X = V * E^-1 * U.t
     return V.T.dot(E_inversed).dot(U.T)

def solve_eq(A: np.array):
    # need to solve Ax=0
    # The vectors which Ax=0 are in the null space of A. 
    # The null space of A corresponds to zero singular values.
    U,E,V = npl.svd(A)
    null_mask = E < 1e-6
    x = V[null_mask]
    # if we check A.dot(x.T) for each solution
    # it will output zeros
    return x

def lines_interscet(l1: np.array, l2: np.array):
    ''' takes arrays with coefs and output dot of the intersection'''
    inter_point = np.cross(l1, l2)
    # transfer to homogenious coordinates
    inter_point /= inter_point[2]
    return inter_point
def main():
    print('___SOLUTION___', end='\n\n')

    print('___TASK1___')
    X = np.array([0.5, 2.16506351, 0.4330127, -0.8660254, 
                1.25, 0.25, 0, 0.5, 2.5]).reshape(3,3)
    print(np.round(closest_matrix(X), 3), end='\n\n')

    print('___TASK2___')
    for n in (3, 10):
        A = np.zeros(n*n).reshape(n, n)
        for i in range(1, n+1):
            for j in range(1, n+1):
                create_elem(A, i, j, task='second')
        sol = inverse_matrix(A)
        print(f'dim = {n}, inversed matrix =\n {np.round(sol, 3)}')

    print('___TASK3___')
    A = np.zeros(4*4).reshape(4, 4)
    for i in range(1, 4+1):
        for j in range(1, 4+1):
            create_elem(A, i, j, task='third')
    print(np.round(solve_eq(A), 3))

    print('___TASK4___')
    l1 = np.random.rand(3)*10
    l2 = np.random.rand(3)*10
    print(np.round(lines_interscet(l1,l2), 3))

main()
