import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat

N = 2000
NperC = 2000/4
Hd = 500
# initially form a clusters in 2D PCA space
sig = np.array([[0.8,0],[0,0.8]])
mu1 = np.array([2, 2])
mu2 = np.array([-2, 2])
mu3 = np.array([2, -2])
mu4 = np.array([-2, -2])

c1 = np.random.multivariate_normal(mu1, sig, int(NperC))
c2 = np.random.multivariate_normal(mu2, sig, int(NperC))
c3 = np.random.multivariate_normal(mu3, sig, int(NperC))
c4 = np.random.multivariate_normal(mu4, sig, int(NperC))
loadings = np.zeros([N, 2])
labels = np.zeros([N, 4])
loadings[:Hd, ...] = c1
loadings[Hd:Hd*2, ...] = c2
loadings[Hd*2:Hd*3, ...] = c3
loadings[Hd*3:, ...] = c4
labels[:Hd, ...] = repmat(np.array([1, 0,  0, 0]), Hd, 1)
labels[Hd:2*Hd, ...] = repmat(np.array([0, 1,  0, 0]), Hd, 1)
labels[2*Hd:3*Hd, ...] = repmat(np.array([0, 0,  1, 0]), Hd, 1)
labels[3*Hd:, ...] = repmat(np.array([0, 0,  0, 1]), Hd, 1)
# Construct as random PC (v1 and v2 are the eigenvectors)
v1 = np.random.randn(Hd, 1)
v2 = np.ones([Hd, 1])
S = np.sum(v1[1:])
v2[0] = -1*S / v1[0]
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)
mu = 10*np.random.randn(Hd, 1)
l1 = 1000
l2 = 100
lam = np.array([np.sqrt(l1), np.sqrt(l2)])
L = repmat(lam, N, 1)
V = np.zeros([Hd, 2])
V[..., 0] = v1[..., 0]
V[..., 1] = v2[..., 0]

ActualData = repmat(mu.T, N, 1) + np.matmul((L*loadings), V.T)
np.save('Data.npy', ActualData)
np.save('ActualLoadings.npy', loadings)
np.save('labels.npy', labels)