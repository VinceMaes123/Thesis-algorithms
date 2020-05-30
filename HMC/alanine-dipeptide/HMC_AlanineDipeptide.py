#Script that runs the HMC method on alanine-dipeptide

#TUNING OF THE METHOD
#For a small amount of samples (eg N = 1e4):
#1) take number of steps = 1 and set epsilon at a high order. The method becomes unstable for large epsilon, so we get
#bad proposals and everything is rejected (distribution is one line). Decrease the order of the stepsize until not
#all proposals are rejected. This is the largest stepsize we can have that keeps the method stable, so this is a good
#stepsize.
#2) start increasing the number of steps L (so also total pathlength epsilon*L between two consecutive samples) until
#The whole distribution is covered with samples. This is the minimum pathlength between samples that allows for an
#efficient sampling of the distribution.
import scipy
from scipy import integrate
import math
import matplotlib.pyplot as plt
import HMC_Stepper_AlanineDipeptide as HMC
import HMC_userDefinedExpressionsAlanineDipeptide as UDE

#LEAPFROG METHOD: choose steplength epsilon
epsilon = 1.0e-2
#LEAPFROG METHOD: choose number of steps L
L = 3
#choose initial condition
q = scipy.array([1.53,1.53,1.335,1.335,1.335,1.335,(113.9 / 360.0) * 2.0 * math.pi,(113.9 / 360.0) * 2.0 * math.pi,
                 (113.9 / 360.0) * 2.0 * math.pi,(117.6 / 360.0) * 2.0 * math.pi,(117.6 / 360.0) * 2.0 * math.pi,
                 0.0,0.0]) #these are all equilibrium values

#choose number of samples N
N = int(1e5)

#HMC METHOD
#initialize micro state
print("starting HMC method...")
CC1 = scipy.zeros(N)
CC2 = scipy.zeros(N)
CN1 = scipy.zeros(N)
CN2 = scipy.zeros(N)
CN3 = scipy.zeros(N)
CN4 = scipy.zeros(N)
CCN1 = scipy.zeros(N)
CCN2 = scipy.zeros(N)
CCN3 = scipy.zeros(N)
CNC1 = scipy.zeros(N)
CNC2 = scipy.zeros(N)
phi = scipy.zeros(N)
psi = scipy.zeros(N)

for n in range(0,N): #exectue loop N times
    q = HMC.HMC_Stepper(epsilon,L,q)
    CC1[n] = q[0]
    CC2[n] = q[1]
    CN1[n] = q[2]
    CN2[n] = q[3]
    CN3[n] = q[4]
    CN4[n] = q[5]
    CCN1[n] = q[6]
    CCN2[n] = q[7]
    CCN3[n] = q[8]
    CNC1[n] = q[9]
    CNC2[n] = q[10]
    phi[n] = q[11]
    psi[n] = q[12]
print("computations done!")

#the correct distribution
xminPsi = -1.05
xmaxPsi = 1.05
scaling_Psi = integrate.quad(UDE.mu_0,xminPsi,xmaxPsi)
scaling_Psi = scaling_Psi[0]
x_coordinate_Psi = scipy.linspace(xminPsi,xmaxPsi,1000)
y_coordinate_Psi = scipy.zeros(scipy.size(x_coordinate_Psi))
for index in range(0,len(x_coordinate_Psi)):
    y_coordinate_Psi[index] = (1.0/scaling_Psi)*UDE.mu_0(x_coordinate_Psi[index])



plt.figure()
plt.hist(psi,bins=1000,range=[xminPsi,xmaxPsi],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_Psi,y_coordinate_Psi)
plt.xlabel(r'Torsion Angle $\psi$')
plt.show()


UDE.makePlot(phi,-0.25,0.25,UDE.phi_0,r'Torsion angle $\phi$')
UDE.makePlot(CC1,1.515-0.05,1.515+0.05,UDE.CC_0,r'Bond length CC')
UDE.makePlot(CN1,1.335-0.05,1.335+0.05,UDE.CN_0,r'Bond length CN')
UDE.makePlot(CCN1,(113.9 / 360.0) * 2.0 * math.pi-0.10,(113.9 / 360.0) * 2.0 * math.pi+0.10,UDE.CCN_0,r'Bond angle CCN')
UDE.makePlot(CNC1,(117.6 / 360.0) * 2.0 * math.pi-0.10,(117.6 / 360.0) * 2.0 * math.pi+0.10,UDE.CNC_0,r'Bond angle CNC')






