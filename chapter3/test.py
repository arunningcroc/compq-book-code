import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import math
N=100
endpoint = 30
Npoints = 3000
Norb = 2
r = np.linspace(1e-3,endpoint,Npoints)
dx = r[1]-r[0]
def integrate(f):
    return np.sum(f)*dx
def rhodiff(prevrho,rho):
    return integrate((prevrho-rho)**2)
def slater(order,alpha):
    N = (2*alpha)**order*np.sqrt(2*alpha/math.factorial(2*order))
    func = N*r[:]**(order-1)*np.exp(-alpha*r[:])
    return r[:]*func
def slater_derivative(order,alpha):
    N = (2*alpha)**order*np.sqrt(2*alpha/math.factorial(2*order))
    dfunc = N*(-alpha*np.exp(-alpha*r[:])*r[:] + np.exp(-alpha*r[:]))
    ddfunc = N*(alpha**2*np.exp(-alpha*r[:])*r[:] -alpha*np.exp(-alpha*r[:]) -alpha*np.exp(-alpha*r[:]) )
    return -0.5*ddfunc
def basis_function(i):
    alphastart = 0.3
    alphastep = 0.05
    if i<100:
        return slater(1,alphastart+i*alphastep)

def basis_derivative(i):
    alphastart = 0.3
    alphastep = 0.05
    if i<100:
        return slater_derivative(1,alphastart+i*alphastep)

def normalized_orbital_density(vec):
    phi = np.zeros(Npoints)
    for i in range(N):
        phi[:] += vec[i]*basis_function(i)
    orbdens = phi[:]**2
    normorbdens = orbdens/(integrate(orbdens))
    return 2*normorbdens
def get_density(vec):
    rho = np.zeros(Npoints)
    for i in range(Norb):
        rho +=  normalized_orbital_density(vec[:,i])
    return rho/(4*np.pi*r[:]**2)
def get_hartree(rho):
    densi = interp1d(r,rho)
    def integrand(x):
        return densi(x)*x*4*np.pi
    
    start = quad(integrand,r[0],endpoint)[0]
    def dV_dr(x,V):
        return V[1],-4*np.pi*densi(x)-2/x*V[1]
    sol = solve_ivp(dV_dr, [r[0],endpoint],[start,0]\
                ,t_eval = r,method="LSODA")
    return sol.y[0]

def get_xc(rho):
    #The commented part is the correlation potential. You can uncomment it
    #and see what happens.
    '''y0 = -0.10498
    b = 3.72744
    c = 12.9352
    A = 0.0621814

    rs = (3/(4*np.pi*rho[:]))**(1.0/3)
    Q = np.sqrt(4*c - b**2)
    y = np.sqrt(rs)
    def get_Y(y,b,c):
        return y**2 + b*y + c
    ec = A/2 * (np.log(y**2/get_Y(y, b, c)) + 2*b/Q * np.arctan(Q/(2*y+b))  \
   - b*y0/get_Y(y0, b, c) * ( \
            np.log((y-y0)**2 / get_Y(y, b, c)) \
            + 2*(b+2*y0) / Q * np.arctan(Q/(2*y+b)) \
          ) )
    Vc =  ec - A/6 * (c*(y-y0)-b*y0*y)/((y-y0)*get_Y(y, b, c))'''
    return -3/(4*np.pi) * (3*np.pi**2*rho[:])**(1.0/3)# + Vc
def hamiltonian(rho):
    H=np.zeros((N,N))
    Vh = get_hartree(rho)
    Vxc = get_xc(rho)
    Va = -4/r[:]
    vks = Vh+Vxc+Va
    for i in range(N):
        for j in range(i,N):
            H[i,j]+=integrate(basis_function(i)*vks[:]*basis_function(j))
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
def get_initial_potential():
    Z = 4
    x = r[:] * (128*Z/(9*np.pi**2)) ** (1.0/3)
    alpha = 0.7280642371
    beta = -0.5430794693
    gamma = 0.3612163121
    Z_eff = Z * (1 + alpha*np.sqrt(x) + beta*x*np.exp(-gamma*np.sqrt(x)))**2 * \
        np.exp(-2*alpha*np.sqrt(x))
    for i in range(len(Z_eff)):
        if Z_eff[i] < 1:
            Z_eff[i] = 1
    return  -Z_eff / r[:]
overlap = get_overlap()

H = np.zeros((N,N))
vtf = get_initial_potential()
for i in range(N):
    for j in range(i,N):
        H[i,j]+=integrate(basis_function(i)*vtf*basis_function(j))
        H[i,j]+= integrate(basis_function(i)*basis_derivative(j))
        H[j,i] = H[i,j]
energies,vec=eigh(H,b=overlap)
initrho = get_density(vec)
rho = initrho
prevrho = initrho
maxiter = 200
alpha = 0.2
for i in range(maxiter):
    H = hamiltonian(rho)
    energies,vec=eigh(H,b=overlap)
    newrho = get_density(vec[:,0:Norb])

    rhod = rhodiff(prevrho,newrho)
    if rhodiff(prevrho,newrho) < 1e-5:
        print("woo!")
        break

    rho = (1-alpha)*prevrho+alpha*newrho
    prevrho=rho
    print("iteration ",i,rhod,energies[0:2])


plt.plot(r,4*np.pi*r**2*rho)
plt.show()
