import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import eigh
import math
N=50
Npoints = 2000
Norb = 2
r = np.linspace(1e-3,49,Npoints)
dx = r[1]-r[0]
vks = np.loadtxt("vks")
vksi = interp1d(vks[:,0],vks[:,1])
vksf = vksi(r)
def integrate(f):
    return np.sum(f)*dx
def slater(order,alpha):
    N = (2*alpha)**order*np.sqrt(2*alpha/math.factorial(2*order))
    func = N*r[:]**(order-1)*np.exp(-alpha*r[:])
    return r[:]*func
def slater_derivative(order,alpha):
    N = (2*alpha)**order*np.sqrt(2*alpha/math.factorial(2*order))
    dfunc = N*(-alpha*np.exp(-alpha*r[:])*r[:] + np.exp(-alpha*r[:]))
    ddfunc = N*(alpha**2*np.exp(-alpha*r[:])*r[:] -alpha*np.exp(-alpha*r[:]) -alpha*np.exp(-alpha*r[:]) )
    return -0.5*ddfunc
def normalized_orbital_density(vec):
    phi = np.zeros(Npoints)
    for i in range(N):
        phi[:] += vec[i]*basis_function(i)
    orbdens = phi[:]**2
    
    return 2*orbdens
def get_density(vec):
    rho = np.zeros(Npoints)
    for i in range(Norb):
        rho +=  normalized_orbital_density(vec[:,i])
    return rho
def basis_function(i):
    #We set 50 order 1 functions and 50 order 2 functions.
    alphastart = 0.3
    alphastep = 0.1
    if i<100:
        return slater(1,alphastart+i*alphastep)
    #if i>=50:
    #    return slater(2,alphastart+(i-50)*alphastep)
def basis_derivative(i):
    alphastart = 0.3
    alphastep = 0.1
    if i<100:
        return slater_derivative(1,alphastart+i*alphastep)

def hamiltonian():
    H=np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            H[i,j]+=integrate(basis_function(i)*vksf[:]*basis_function(j))
            H[i,j]+= integrate(basis_function(i)*basis_derivative(j))
            H[j,i] = H[i,j]
    return H
def get_overlap():
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            if i==j:
                S[i,i] = 1
            else:
                S[i,j] = integrate(basis_function(i)*basis_function(j))
                S[j,i] = S[i,j]
    return S
H = hamiltonian()
overlap = get_overlap()
print(overlap)
energies,vec=eigh(H,b=overlap)
rho = get_density(vec)
print(energies[0:2])

plt.plot(r[0:500],rho[0:500])
plt.show()