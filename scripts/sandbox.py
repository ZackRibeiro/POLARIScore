from POLARIScore.utils.physics_utils import *

L = 10 #pc
M = 1e5 #solar mass
Cs = 200 #m/s

#cgs
L = L*PC_TO_CM
Cs = Cs*1e2
M = M * 1.988e33

t = L/Cs
print("t="+str(t/(3600*24*365.25*1e6))+" Myrs")

G = 6.674e-8

rho = M / L**3

G_code = 4*np.pi * G * rho * L**2 / Cs**2
print(G_code)
