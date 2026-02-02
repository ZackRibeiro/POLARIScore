from POLARIScore.utils.physics_utils import *
import argparse

"""
Compute common physical quantities in order to run simulations.
"""

L = 5 #pc
Mass = 1e4 #solar mass
T = 30 #K
B = 10 #microgauss
Ms = 5

#############

parser = argparse.ArgumentParser()
parser.add_argument("--L", required=False, default=L, help="Box length")
parser.add_argument("--M", required=False, default=Mass, help="Mass contained in the box")
parser.add_argument("--T", required=False, default=T, help="Temperature")
parser.add_argument("--B", required=False, default=B, help="Magnetic field along z axi (in microgauss)")
parser.add_argument("--Ms", required=False, default=Ms, help="Sonic mach number")
args = parser.parse_args()
L = float(args.L)
T = float(args.T)
B = float(args.B)
Ms = float(args.Ms)
Mass = float(args.M)
print(f"Computed using: L={L}pc | T={T}K | B={B}µG | Ms={Ms} | M={Mass}")
print("---------------------")

kb = 1.380649e-23
Cs = np.sqrt(kb*T/(2.33*1.6735575e-27)) 
print("Cs="+str(Cs)+" m/s")

#cgs
L = L*PC_TO_CM
B *= 1e-6
Cs = Cs*1e2
Mass = Mass * 1.988e33

print("L="+str(L)+" cm")
rho = Mass/(L**3)

t = L/Cs
print("t="+str(t/(3600*24*365.25*1e6))+" Myrs")
print("rho="+str(rho)+" g/cm^3")

G = 6.674e-8
tff = np.sqrt(3*np.pi/(32*rho*G))
print("tff="+str(tff/(3600*24*365.25*1e6))+" Myrs")
tturb = t/2/Ms

G_code = G * t**2*rho
print("G_code="+str(G_code))

B_code = B/(np.sqrt(rho)*Cs)
print("B_code="+str(B_code))

virial = 5*(Ms*Cs)**2*L/2 / (G*Mass)
print("vir="+str(virial))

print("tff/t="+str(tff/t))
print("tturb/t="+str(tturb/t))

# this gives B = 8.49e-6G