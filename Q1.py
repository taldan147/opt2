import numpy as np
from numpy import linalg as LA

A = np.asarray([[4,-1,1], [4,-8,1], [-2,1,5]])
b = np.asarray([[7],[-21],[15]])
x0 = np.asarray([[1],[2],[2]])
def Jacobi(A, b, x0, maxIter):
    D= np.diag(np.asarray(np.diagonal(A)))
    Dinv = LA.inv(D)
    L = np.tril(A) #TODO check if need
    U = np.triu(A)
    x_k = x0
    for k in range(1,maxIter+1):
        x_k = x_k +Dinv@(b-(A@x_k))
        print(x_k)
    return x_k

def GaussSeidel(A, b, x0, maxIter):
    D = np.diag(np.asarray(np.diagonal(A)))
    L = np.tril(A)
    LDinv = LA.inv(L)
    U = np.triu(A)
    x_k = x0
    for k in range(1, maxIter + 1):
        x_k = x_k + LDinv @ (b - (A @ x_k))
        print(x_k)
    return x_k




print(GaussSeidel(A,b,x0, 8))