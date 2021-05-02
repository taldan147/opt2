import numpy as np
from numpy import linalg as LA

from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as pl



def SteepestDescent(A, b, x0, maxIter,w):
    r_k = b - (A @ x0)
    x_k=x0
    norms = []
    convergenceFactors = []
    for k in range(maxIter):
        x_k_old = x_k
        Ar_k =  A @ r_k
        alpha = (np.transpose(r_k) @ Ar_k) / (np.transpose(r_k) @ np.transpose(A) @ Ar_k)
        x_k = x_k + alpha * r_k
        r_k = r_k - alpha * Ar_k
        currNorm = LA.norm(b-(A @ x_k), 2)
        norms.append(currNorm)
        convergenceFactors.append(currNorm / LA.norm((A @ x_k_old) - b, 2))
    return x_k, norms, convergenceFactors


A = np.asarray([[5,4,4,-1,0], [3,12,4,-5,-5], [-4,2,6,0,3], [4,5,-7,10,2], [1,2,5,3,10]])
b = np.asarray([[1,1,1,1,1]])
b = np.transpose(b)
x0 = np.asarray([[0,0,0,0,0]])
x0 = np.transpose(x0)

x_SD, norms_SD, cf_SD = SteepestDescent(A, b, x0, 50,1)

pl.semilogy(norms_SD, label = 'Steepest descend')
pl.xlabel("k'th iteration")
pl.ylabel(r'${||b-Ax^{(k)}||_2}$')
pl.legend()
pl.show()