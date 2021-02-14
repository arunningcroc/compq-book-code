import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,4,1000)
dx = x[1]-x[0]
kinetic_energy = -0.5/dx**2 \
                * (np.diagflat(-2*np.ones(1000))\
                +np.diagflat(np.ones(999),-1)\
                +np.diagflat(np.ones(999),1))
kinetic_energy[0,999] = -0.5/dx**2
kinetic_energy[999,0] = -0.5/dx**2
energies,vec = np.linalg.eigh(kinetic_energy)
print(energies[0:4])
plt.plot(x,vec[:,0])
plt.plot(x,vec[:,1])
plt.plot(x,vec[:,2])
plt.show()