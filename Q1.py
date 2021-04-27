import numpy as np
from numpy import linalg as LA

from scipy.sparse import random
import scipy.sparse as sparse

import math


def Jacobi(A, b, x0, maxIter, w):
    D= np.diag(np.asarray(np.diagonal(A)))
    Dinv = LA.inv(D)
    L = np.tril(A) #TODO check if need
    U = np.triu(A)
    x_k = x0
    for k in range(1,maxIter+1):
        x_k = x_k + w * Dinv@(b-(A@x_k))
    return x_k

def GaussSeidel(A, b, x0, maxIter):
    n = len(b)
    x_k = x0
    for k in range(1, maxIter + 1):
        for i in range(0, n):
            res = b[i]
            for j in range(0, n):
                if (i != j):
                    res = res - A[i][j] * x_k[j]
            x_k[i] = res/A[i][i]
    return x_k

def SteepestDescent(A, b, x0, maxIter):
    r_k = b - (A @ x0)
    x_k=x0
    for k in range(maxIter):
        Ar_k =  A @ r_k
        alpha = (np.transpose(r_k) @ r_k) / (np.transpose(r_k) @ Ar_k)
        x_k = x_k + alpha * r_k
        r_k = r_k - alpha * Ar_k
    return x_k

def ConjugateGradient(A, b, x0, maxIter):
    r_k = b - (A @ x0)
    p_k = b - (A @ x0)
    x_k = x0
    for k in range(maxIter):
        Ap_k = A @ p_k
        alpha = (np.transpose(r_k) @ r_k) / (np.transpose(p_k) @ Ap_k)
        x_k = x_k + alpha * p_k
        r_k_new = r_k - alpha * Ap_k
        beta = -1 * ((np.transpose(r_k_new) @ r_k_new) / (np.transpose(r_k) @ r_k))
        p_k = r_k_new + beta * p_k
        r_k = r_k_new
    return x_k

n = 256
# A = random(n, n, 5 / n,'array', dtype=float)
# A = np.asarray(A)
# v = np.random.rand(n)
# v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'array')
# A = A.transpose() @ v @ A + 0.1*sparse.eye(n)

A = random(n, n, 5 / n, dtype=float)
v = np.random.rand(n)
v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr').toarray()
A = A.transpose() * v * A + 0.1*sparse.eye(n)

D= np.diag(np.asarray(np.diagonal(A)))
T = np.eye(n) - (0.001* LA.inv(D) @ A)
norm = LA.norm(T,2)
print(norm)

x0 = np.zeros(n)
b = np.random.rand(n)
A=np.asarray(A)
print(A)

ans = Jacobi(A,b,x0,100,0.5)
# print("Jacobi\n", ans)
# print("Gauss-Seidel\n", GaussSeidel(A,b,x0,100))
# print("SteepestDescent\n", SteepestDescent(A,b,x0,100))
# print("Conjugate Gradient\n", ConjugateGradient(A,b,x0,100))