import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
N=4001
Norb = 2
r = np.linspace(1e-3,10,N)
dx = r[1]-r[0]
vks = np.loadtxt("vks")
vksi = interp1d(vks[:,0],vks[:,1])
vksf = vksi(r)
def loggrid(Npoints,rmin,rmax,a):
    r = np.zeros(Npoints)
    Dr = np.zeros(Npoints)
    beta = np.log(a)/(Npoints-2)
    alpha = (rmax-rmin)/(np.exp(beta*(Npoints-1))-1)
    for i in range(Npoints):
        r[i] = alpha * (np.exp(beta*(i))-1 )+rmin
        Dr[i]= alpha * beta * np.exp(beta*(i))
    return (r,Dr)
def normalized_orbital_density(vec,Dr):
    integral = np.sum(vec**2*Dr)
    return 2*vec**2/integral

def get_density(vec,Dr):
    newdensity = np.zeros(N)
    for i in range(Norb):
        orbrho=normalized_orbital_density(vec[:,i],Dr)
        newdensity+=orbrho

    return newdensity
def hamiltonian(rho):
    D = np.diagflat(-np.ones(N-2),2) + np.diagflat(-np.ones(N-2),-2)\
        +np.diagflat(16*np.ones(N-1),1) + np.diagflat(16*np.ones(N-1),-1)\
        +np.diagflat(-30*np.ones(N))
    D = -0.5*D/(12*dx**2)

    H = D + np.diagflat(vksf)
    return H
def hamiltonian_log(rho,r,Dr):
    rowu = -np.ones(N-2)/(Dr[1:N-1]*12)
    rowl = 8*np.ones(N-1)/(Dr[1:N]*12)
    D = np.diagflat(-rowu,2)+np.diagflat(rowu,-2)+np.diagflat(-rowl,-1)+np.diagflat(rowl,1)
    D2 = D.dot(D)
    return -0.5*D+np.diagflat(vksi(r))
def hamiltonian_log_3p(rho,r,Dr):
    diag = -np.ones(N)/Dr[:]
    above = np.ones(N-1)/Dr[1:N]
    D = np.diagflat(diag)+np.diagflat(above,1)
    D2 = D.dot(-D.T)
    D2[-1,-1] = D[0,0]
     
    return -0.5*D2+np.diagflat(vksi(r))
r,Dr = loggrid(N,1.0e-7,15,2.7e6)
H = hamiltonian_log_3p(np.zeros(10),r,Dr)
en,vec = np.linalg.eigh(H)

rho = get_density(vec,Dr)
print(sum(rho*Dr))
plt.plot(r,rho)
plt.show()