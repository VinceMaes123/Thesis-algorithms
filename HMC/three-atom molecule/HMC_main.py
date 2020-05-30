#Script that runs the HMC method on a given problem: the 3-atom molecule

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
import HMC_Stepper as HMC
import HMC_userDefinedExpressions as UDE


#LEAPFROG METHOD: choose steplength epsilon
epsilon = 1.0e-3
#LEAPFROG METHOD: choose number of steps L
L = 10
#choose initial condition
q = scipy.array([1.0, math.cos(math.pi/2.0-0.3838), math.sin(math.pi/2.0-0.3838)])
#choose number of samples N
N = int(1e5)

#HMC METHOD
#initialize micro state
print("starting HMC method...")
eta = scipy.zeros(N) #initial eta is not saved in eta
xas = scipy.zeros(N) #initial x_a is not saved in xas
rcs = scipy.zeros(N) #initial r_c is not saved in rcs
xcs = scipy.zeros(N) #initial x_c is not saved in xcs
ycs = scipy.zeros(N) #initial y_c is not saved in ycs

for n in range(0,N): #execute loop N times
    q = HMC.HMC_Stepper(epsilon,L,q)
    eta[n] = UDE.restriction(q)
    xas[n] = q[0]
    rcs[n] = math.sqrt(q[1]**2 + q[2]**2)
    xcs[n] = q[1]
    ycs[n] = q[2]
print("computations done!")

#the correct distribution
scaling_eta = integrate.quad(UDE.mu_0,0.0,3.0)
scaling_eta = scaling_eta[0]
x_coordinate_eta = scipy.linspace(0.0,3.0,1000)
y_coordinate_eta = scipy.zeros(scipy.size(x_coordinate_eta))
for index in range(0,len(x_coordinate_eta)):
    y_coordinate_eta[index] = (1.0/scaling_eta)*UDE.mu_0(x_coordinate_eta[index])

scaling_c = integrate.quad(UDE.distplot,0.8,1.2,args=(UDE.epsilon()))
scaling_c = scaling_c[0]
x_coordinate_c = scipy.linspace(0.8,1.2,1000)
y_coordinate_c = scipy.zeros(scipy.size(x_coordinate_c))
for index in range(0,len(x_coordinate_c)):
    y_coordinate_c[index] = (1.0/scaling_c)*math.exp((-0.5)*(x_coordinate_c[index]-1.00)**2/UDE.epsilon())

xmax = 1.0+math.sqrt((UDE.epsilon())/(1.0e-2)) #These limits yield nice figures for the xas and rcs distributions
xmin = 1.0-math.sqrt((UDE.epsilon())/(1.0e-2)) #Limits scale with sqrt(eps) because that is the std of the distribution

plt.figure()
plt.hist(eta,bins=1000,range=[0.0,3.0],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_eta,y_coordinate_eta)
#plt.title("eta")
plt.xlabel(r'$\theta$')
plt.show()

# plt.figure()
# plt.hist(xas,bins=1000,range=[xmin,xmax],density=1)
# plt.plot(x_coordinate_c,y_coordinate_c)
# plt.xlim((xmin,xmax))
# plt.title("x_a")
# plt.show()
#
# plt.figure()
# plt.hist(xcs,bins=1000,density=1)
# plt.title("x_c")
# plt.show()
#
# plt.figure()
# plt.hist(ycs,bins=1000,density=1)
# plt.title("y_c")
# plt.show()
#
# plt.figure()
# plt.hist(rcs,bins=1000,range=[xmin,xmax],density=1)
# plt.plot(x_coordinate_c,y_coordinate_c)
# plt.xlim((xmin,xmax))
# #plt.title("r_c")
# plt.show()
# #plt.savefig('test.pdf') #save data to a pdf
# #scipy.io.savemat('mM_MCMC',{'xas':xas,'xcs':xcs,'ycs':ycs}) #save to .mat file that can be interpreted by MATLAB






