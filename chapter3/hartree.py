import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.integrate import quad
dens = np.loadtxt("results")
r = np.linspace(dens[0,0],50,2000)
dr = r[1]-r[0]

densi =  interp1d(dens[:,0],dens[:,1])
def integrand(x):
    return densi(x)*x*4*np.pi

start = quad(integrand,dens[0,0],50)[0]
print(start)
def dV_dr(x,V):
    return V[1],-4*np.pi*densi(x)-2/x*V[1]

sol = solve_ivp(dV_dr, [dens[0,0],10],[start,0],method="LSODA")

plt.grid()
plt.plot(sol.t,sol.y[0])
plt.show()