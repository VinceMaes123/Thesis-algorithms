#Script that runs the HMC method on butane

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
import HMC_Stepper_Butane as HMC
import HMC_userDefinedExpressionsButane as UDE

#LEAPFROG METHOD: choose steplength epsilon
epsilon = 1.0e-2
#LEAPFROG METHOD: choose number of steps L
L = 5
#choose initial condition
q = scipy.array([1.53,1.53,1.53,(112.0/360.0)*2.0*math.pi,(112.0/360.0)*2.0*math.pi,0.0]) #these are all equilibrium values

#choose number of samples N
N = int(1e5)

#HMC METHOD
#initialize micro state
print("starting HMC method...")
BL1 = scipy.zeros(N) #Bond lengths
BL2 = scipy.zeros(N)
BL3 = scipy.zeros(N)
BA1 = scipy.zeros(N) #Bond angles
BA2 = scipy.zeros(N)
TA = scipy.zeros(N) #Torsion angle

for n in range(0,N): #execute loop N times
    q = HMC.HMC_Stepper(epsilon,L,q)
    BL1[n] = q[0]
    BL2[n] = q[1]
    BL3[n] = q[2]
    BA1[n] = q[3]
    BA2[n] = q[4]
    TA[n] = q[5]
print("computations done!")

#the correct distribution
xminTA = -3.5
xmaxTA = 3.5
scaling_TA = integrate.quad(UDE.mu_0,xminTA,xmaxTA)
scaling_TA = scaling_TA[0]
x_coordinate_TA = scipy.linspace(xminTA,xmaxTA,1000)
y_coordinate_TA = scipy.zeros(scipy.size(x_coordinate_TA))
for index in range(0,len(x_coordinate_TA)):
    y_coordinate_TA[index] = (1.0/scaling_TA)*UDE.mu_0(x_coordinate_TA[index])



plt.figure()
plt.hist(TA,bins=1000,range=[xminTA,xmaxTA],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_TA,y_coordinate_TA)
plt.xlabel('Torsion Angle')
plt.show()


UDE.makePlot(BL1,1.53-0.10,1.53+0.10,UDE.CC_0,r'Bond length CC')
UDE.makePlot(BA1,(112.0 / 360.0) * 2.0 * math.pi-0.20,(112.0 / 360.0) * 2.0 * math.pi+0.20,UDE.CCC_0,r'Bond angle CCC')






