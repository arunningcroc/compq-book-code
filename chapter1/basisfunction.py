import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np
#Number of barriers
nbar = 10
a = 1
b = 1/6
V0 = 100
xi = np.zeros(nbar)
L=10
npoints = 100
H = np.zeros((npoints,npoints))
def fnm(x,n,m):
	if n==m:
		return x/L - \
		np.sin(2*np.pi*n*x/L)/(2*np.pi*n)
	else:
		return np.sin((m-n)*np.pi*x/L)/\
		(np.pi*(m-n))- \
		np.sin((m+n)*np.pi*x/L)\
		/(np.pi*(m+n))
def hnm(x,n,m):
	return fnm(x+b/2,n,m) - fnm(x-b/2,n,m)        
for i in range(nbar):
	xi[i] = (-0.5)*a + (i+1) * a

for i in range(npoints):
	for j in range(i,npoints):
		for k in range(nbar):
			H[i,j] += V0\
			         *hnm(xi[k],i+1,j+1)

		if i == j:
			H[i,j] += np.pi**2\
			          *(i+1)**2/(L**2)
		H[j,i] = H[i,j]

e, vec = eigh(H)
x = np.zeros(30)
for i in range(30):
	x[i] = (i+1)*np.pi/L
plt.xlabel("Wave number k of solution")
plt.ylabel("Energy")
plt.scatter(x/np.pi,e[0:30]/np.pi**2)
plt.show()