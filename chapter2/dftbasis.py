import numpy as np
import matplotlib.pyplot as plt

L=8.0
Np=1000
Norb = 5
lamb = 50
maxiter = 200
Nbasis = 100
x = np.linspace(0,L,Np)
dx = x[1]-x[0]
initrho = np.zeros(Np)

for i in range(1,Norb+1):
    initrho[:]+=2*2.0/L*np.sin(i*x[:]*np.pi/L)**2.0
print(np.sum(initrho)*dx)

def integrate(f):
    return np.sum(f)*dx
def basis_function(i):
    return np.sqrt(2.0/L)*np.sin((i+1)*x[:]*np.pi/L)
def normalized_orbital_density(vec):
    phi = np.zeros(Np)
    for i in range(Nbasis):
        phi[:] += vec[i]*basis_function(i)
    orbdens = phi[:]**2
    
    return 2*orbdens
def hamiltonian(rho):
    H = np.zeros((Nbasis,Nbasis))
    for i in range(Nbasis):
        H[i,i] = ((i+1)**2*np.pi**2)/(2*L)

    for i in range(Nbasis):
        for j in range(Nbasis):
            pot = lamb*0.25*rho[:]
            b1 = basis_function(i)
            b2 = basis_function(j)
            H[i,j] += integrate(b1*pot*b2)

    return H
def get_density(vec):
    newdensity = np.zeros(Np)
    for i in range(Norb):
        orbrho=normalized_orbital_density(vec[:,i])
        newdensity+=orbrho
    return newdensity
def rhodiff(prevrho,rho):
    return integrate((prevrho-rho)**2)
rho = initrho
prevrho = initrho
alpha = 0.1


for i in range(maxiter):
    H = hamiltonian(rho)
    #print(H)
    energies,vec=np.linalg.eigh(H)

    newrho = get_density(vec[:,0:Norb])
    
    if rhodiff(prevrho,newrho) < 1e-5:
        print("woo!")
        break
    rho = (1-alpha)*prevrho+alpha*newrho
    prevrho=rho
    print("iteration ",i,energies[0])
print(dx*np.sum(rho))
plt.plot(initrho)
plt.plot(rho)
plt.show()
    