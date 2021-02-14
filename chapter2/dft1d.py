import numpy as np
import matplotlib.pyplot as plt

L=8.0
Np=1000
Norb = 5
lamb = 8
maxiter = 200

x = np.linspace(0,L,Np)
dx = x[1]-x[0]
initrho = np.zeros(Np)

for i in range(1,Norb+1):
    initrho[:]+=2*2.0/L*np.sin(i*x[:]*np.pi/L)**2.0
print(np.sum(initrho)*dx)
def normalized_orbital_density(vec):
    integral = np.sum(vec**2)*dx
    return 2*vec**2/integral

def hamiltonian(rho):
    kinetic_energy=(np.diagflat(-2*np.ones(1000))\
                  +np.diagflat(np.ones(999),-1)\
                  +np.diagflat(np.ones(999),1))
    kinetic_energy*=-0.5/dx**2

    vee=lamb/4 * rho[:]
    Hee = np.diagflat(vee)
    H = kinetic_energy+Hee
    return H
def get_density(vec):
    newdensity = np.zeros(Np)
    for i in range(Norb):
        orbrho=normalized_orbital_density(vec[:,i])
        newdensity+=orbrho

    return newdensity
def rhodiff(prevrho,rho):
    integral = np.sum((prevrho-rho)**2)
    integral*= dx
    return integral
rho = initrho
prevrho = initrho
alpha = 0.1
for i in range(maxiter):
    H = hamiltonian(rho)
    energies,vec=np.linalg.eigh(H)
    newrho = get_density(vec[:,0:Norb])
    
    if rhodiff(prevrho,newrho) < 1e-5:
        print("woo!")
        break
    rho = (1-alpha)*prevrho+alpha*newrho
    prevrho=rho
    print("iteration ",i, energies[0])
print(dx*np.sum(rho))
plt.plot(initrho)
plt.plot(rho)
plt.show()
    