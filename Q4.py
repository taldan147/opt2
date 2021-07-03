import numpy as np
from numpy import linalg as LA

from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as pl

from Q1 import Jacobi


A = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,-1,0,0,0,0,0,0], [0,0,-1,5,-1,0,-1,0,-1,-1], [0,0,0,-1,4,-1,-1,-1,0,0],[0,0,0,0,-1,3,-1,-1,0,0,],[0,0,0,-1,-1,-1,5,-1,0,-1],[0,0,0,0,-1,-1,-1,4,0,-1],[0,0,0,-1,0,0,0,0,2,-1],[0,0,0,-1,0,0,-1,-1,-1,4]])


ones = np.asarray([[1,1,1,1,1,1,1,1,1,1]])
b = np.asarray([[1,-1,1,-1,1,-1,1,-1,1,-1]])
b = np.transpose(b)
x0 = np.asarray([[0,0,0,0,0,0,0,0,0,0]])
x0 = np.transpose(x0)

x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi(A,b,x0,200,1,0.000001)
pl.semilogy(norms_Jacobi, label = '4.1 residual')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = '4.1 convergence factor')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.legend()
pl.show()

print(x_Jacobi)
print("b.1 number of iterations:", len(cf_Jacobi))
print("b.1 convergence factor",cf_Jacobi[(len(cf_Jacobi)-1)])

def Jacobi2(A, b, x0, maxIter, w, epsilon, M):
    Minv = LA.inv(M)
    x_k = x0
    norms = []
    bNorm = LA.norm(b)
    convergenceFactors = []
    for k in range(1,maxIter+1):
        x_k_new = x_k + w * Minv@(b-(A@x_k))
        currNorm = LA.norm((A @ x_k_new) - b, 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm/ LA.norm((A @ x_k) - b, 2))
        x_k = x_k_new
        if currNorm /bNorm < epsilon:
            break
    return x_k, norms, convergenceFactors

M = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,0,0,0,0,0,0,0], [0,0,0,5,-1,0,-1,0,-1,-1], [0,0,0,-1,4,-1,-1,-1,0,0],[0,0,0,0,-1,3,-1,-1,0,0,],[0,0,0,-1,-1,-1,5,-1,0,-1],[0,0,0,0,-1,-1,-1,4,0,-1],[0,0,0,-1,0,0,0,0,2,-1],[0,0,0,-1,0,0,-1,-1,-1,4]])
x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi2(A,b,x0,200,0.7,0.000001, M)
pl.semilogy(norms_Jacobi, label = '4.2 residual')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = '4.2 convergance factor')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.legend()
pl.show()

print(x_Jacobi)
print("b.2 number of iterations:", len(cf_Jacobi))
print("b.2 convergence factor",cf_Jacobi[(len(cf_Jacobi)-1)])

L1 = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,0,0,0,0,-1,0,0], [0,0,0,4,-1,-1,-1,0,0,-1], [0,0,0,-1,4,-1,-1,-1,0,0],[0,0,0,-1,-1,3,-1,0,0,0],[0,0,0,-1,-1,-1,5,-1,0,-1],[0,0,-1,0,-1,0,-1,5,-1,-1],[0,0,0,0,0,0,0,-1,2,-1],[0,0,0,-1,0,0,-1,-1,-1,4]])


# {1,2,3}, {4,5,6,7}, {8,9,10}
M1 = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,0,0,0,0,0,0,0], [0,0,0,5,-1,0,-1,0,0,0], [0,0,0,-1,4,-1,-1,0,0,0],[0,0,0,0,-1,3,-1,0,0,0,],[0,0,0,-1,-1,-1,5,0,0,0],[0,0,0,0,0,0,0,4,0,-1],[0,0,0,0,0,0,0,0,2,-1],[0,0,0,0,0,0,0,-1,-1,4]])

#  {1,2,3}, {4,9,10}, {5,6,7,8}
M2 = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,0,0,0,0,0,0,0], [0,0,0,4,-1,-1,-1,0,0,0], [0,0,0,-1,4,-1,-1,0,0,0],[0,0,0,0-1,-1,3,-1,0,0,0],[0,0,0,-1,-1,-1,5,0,0,0],[0,0,0,0,0,0,0,5,-1,-1],[0,0,0,0,0,0,0,-1,2,-1],[0,0,0,0,0,0,0,-1,-1,4]])

#{1,2,3}, {4,7,9,10}, {5,6,8}
M3 = np.asarray([[2,-1,-1,0,0,0,0,0,0,0], [-1,2,-1,0,0,0,0,0,0,0], [-1,-1,3,0,0,0,0,0,0,0],  [0,0,0,4,-1,-1,0,0,0,0,],[0,0,0,-1,4,-1,0,0,0,0],[0,0,0,-1,-1,3,0,0,0,0],[0,0,0,0,0,0,5,-1,0,-1],[0,0,0,0,0,0,-1,5,-1,-1],[0,0,0,0,0,0,0,-1,2,-1],[0,0,0,0,0,0,-1,-1,-1,4]])
x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi2(A,b,x0,200,0.65,0.000001, M1)
pl.semilogy(norms_Jacobi, label = 'M1 residual')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = 'M1 convergence factor')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.legend()
pl.show()

cg = LA.eigvals(np.eye(10)- 0.65 * (LA.inv(M1) @ A))
print("cg1:", cg)
print("b.3 M1 number of iterations:", len(cf_Jacobi))
print("b.3 M1 convergence factor",cf_Jacobi[(len(cf_Jacobi)-1)])


x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi2(L1,b,x0,200,0.65,0.000001, M2)
pl.semilogy(norms_Jacobi, label = 'M2 residual')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = 'M2 convergence factor')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.legend()
pl.show()

cg = LA.eigvals(np.eye(10)- 0.65 * (LA.inv(M2) @ L1))
print("cg2:", cg)
print("b.3 M2 number of iterations:", len(cf_Jacobi))
print("b.3 M2 convergence factor",cf_Jacobi[(len(cf_Jacobi)-1)])


x_Jacobi, norms_Jacobi , cf_Jacobi = Jacobi2(L1,b,x0,200,0.65,0.000001, M3)
pl.semilogy(norms_Jacobi, label = 'M3 residual')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||Ax^{(k)}-b||_2}$')
pl.legend()
pl.show()

pl.semilogy(cf_Jacobi, label = 'M3 convergence factor')
pl.xlabel("k'th iteration")
pl.ylabel(r'$\frac{||Ax^{(k)}-b||_2}{||Ax^{(k-1)}-b||_2}$')
pl.legend()
pl.show()

cg = LA.eigvals(np.eye(10)- 0.65 * (LA.inv(M3) @ L1))
print("cg3:", cg)
print("b.3 M3 number of iterations:", len(cf_Jacobi))
print("b.3 M3 convergence factor",cf_Jacobi[(len(cf_Jacobi)-1)])

