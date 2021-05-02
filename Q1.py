import numpy as np
from numpy import linalg as LA

from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as pl

import math


def Jacobi(A, b, x0, maxIter, w):
    D= np.diag(np.asarray(np.diagonal(A)))
    Dinv = LA.inv(D)
    L = np.tril(A) #TODO check if need
    U = np.triu(A)
    x_k = x0
    norms = []
    convergenceFactors = []
    for k in range(1,maxIter+1):
        x_k_new = x_k + w * Dinv@(b-(A@x_k))
        currNorm = LA.norm((A @ x_k_new) - b, 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm/ LA.norm((A @ x_k) - b, 2))
        x_k = x_k_new
    return x_k, norms, convergenceFactors

def GaussSeidel(A, b, x0, maxIter,w):
    n = len(b)
    x_k = x0
    norms = []
    convergenceFactors = []
    for k in range(1, maxIter + 1):
        x_k_old = x_k
        for i in range(0, n):
            res = b[i]
            for j in range(0, n):
                if (i != j):
                    res = res - A[i][j] * x_k[j]
            x_k[i] = res/A[i][i]
        currNorm = LA.norm((A @ x_k) - b, 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm / LA.norm((A @ x_k_old) - b, 2))
    return x_k, norms, convergenceFactors

def SteepestDescent(A, b, x0, maxIter,w):
    r_k = b - (A @ x0)
    x_k=x0
    norms = []
    convergenceFactors = []
    for k in range(maxIter):
        x_k_old = x_k
        Ar_k =  A @ r_k
        alpha = (np.transpose(r_k) @ r_k) / (np.transpose(r_k) @ Ar_k)
        x_k = x_k + alpha * r_k
        r_k = r_k - alpha * Ar_k
        currNorm = LA.norm((A @ x_k) - b, 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm / LA.norm((A @ x_k_old) - b, 2))
    return x_k, norms, convergenceFactors

def ConjugateGradient(A, b, x0, maxIter,w):
    r_k = b - (A @ x0)
    p_k = b - (A @ x0)
    x_k = x0
    norms = []
    convergenceFactors = []
    for k in range(maxIter):
        x_k_old = x_k
        Ap_k = A @ p_k
        alpha = (np.transpose(r_k) @ r_k) / (np.transpose(p_k) @ Ap_k)
        x_k = x_k + alpha * p_k
        r_k_new = r_k - alpha * Ap_k
        beta = -1 * ((np.transpose(r_k_new) @ r_k_new) / (np.transpose(r_k) @ r_k))
        p_k = r_k_new + beta * p_k
        r_k = r_k_new
        currNorm = LA.norm((A @ x_k) - b, 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm / LA.norm((A @ x_k_old) - b, 2))
    return x_k, norms, convergenceFactors

n = 256

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

x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi(A,b,x0,100,0.3)
x_GS, norms_GS, cf_GS = GaussSeidel(A, b, x0, 100,1)
x_SD, norms_SD, cf_SD = SteepestDescent(A, b, x0, 100,1)
x_CG, norms_CG, cf_CG = ConjugateGradient(A, b, x0, 100,1)
pl.semilogy(norms_Jacobi, label = 'Jacobi')
pl.semilogy(norms_GS, label = 'Gauss Seidel')
pl.semilogy(norms_SD, label = 'Steepest descend')
pl.semilogy(norms_CG, label = 'Conjugate Gradient')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = 'Jacobi')
pl.semilogy(cf_GS, label = 'Gauss Seidel')
pl.semilogy(cf_SD, label = 'Steepest descend')
pl.semilogy(cf_CG, label = 'Conjugate Gradient')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.show()


# print("Jacobi\n", ans)
# print("Gauss-Seidel\n", GaussSeidel(A,b,x0,100))
# print("SteepestDescent\n", SteepestDescent(A,b,x0,100))
# print("Conjugate Gradient\n", ConjugateGradient(A,b,x0,100))