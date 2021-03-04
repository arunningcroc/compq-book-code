import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.integrate import quad
dens = np.loadtxt("results")

densi =  interp1d(dens[:,0],dens[:,1])
def integrand(x):
    return densi(x)*x*4*np.pi

start = quad(integrand,dens[0,0],50)[0]
def dV_dr(x,V):
    return V[1],-4*np.pi*densi(x)-2/x*V[1]

sol = solve_ivp(dV_dr, [dens[0,0],40],[start,0]\
                ,method="LSODA")

hartree = np.loadtxt("hartree")
harti = interp1d(hartree[:,0],hartree[:,1])
plt.grid()
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,harti(sol.t))
plt.show()