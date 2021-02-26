import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
N=2000
Norb = 2
r = np.linspace(1e-3,5,N)
dx = r[1]-r[0]
vks = np.loadtxt("vks")
vksi = interp1d(vks[:,0],vks[:,1])
vksf = vksi(r)

def normalized_orbital_density(vec):
    integral = np.sum(vec**2)*dx
    return 2*vec**2/integral

def get_density(vec):
    newdensity = np.zeros(N)
    for i in range(Norb):
        orbrho=normalized_orbital_density(vec[:,i])
        newdensity+=orbrho

    return newdensity
D = np.diagflat(-np.ones(N-2),2) + np.diagflat(-np.ones(N-2),-2)\
    +np.diagflat(16*np.ones(N-1),1) + np.diagflat(16*np.ones(N-1),-1)\
    +np.diagflat(-30*np.ones(N))
D = -0.5*D/(12*dx**2)

H = D + np.diagflat(vksf)

en,vec = np.linalg.eigh(H)
rho = get_density(vec)
plt.plot(r,rho)
plt.show()