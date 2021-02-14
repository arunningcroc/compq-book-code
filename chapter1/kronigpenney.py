import numpy as np
import matplotlib.pyplot as plt

V0 = 100
npoints = 1000
V = np.zeros(npoints)
x = np.linspace(0,10,npoints)
dx = x[1]-x[0]
V[40:61] = V0 
V[140:161] = V0
V[240:261] = V0
V[340:361] = V0
V[440:461] = V0
V[540:561] = V0
V[640:661] = V0
V[740:761] = V0
V[840:861] = V0
V[940:961] = V0
kinetic_energy = -0.5/dx**2 \
                * (np.diagflat(-2*np.ones(1000))\
                +np.diagflat(np.ones(999),-1)\
                +np.diagflat(np.ones(999),1))
H = kinetic_energy + np.diagflat(V)

e,vec=np.linalg.eigh(H)


plt.rcParams.update({'font.size': 15})
plt.xlabel("k*L")
plt.ylabel("Energy")
plt.scatter(range(30),e[0:30]/np.pi**2)
plt.show()
