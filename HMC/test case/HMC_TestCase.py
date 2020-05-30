#Script that runs the HMC method on a given testcase.
#The testcase is a 2D normal distribution with variances 1 and 4 (so std 1 and 2), and expected values 2 and 3.
#The distribution can be written as exp( -0.5*(q0-2)^2/1 - 0.5*(q1-3)^2/4 )
#We take V(q) = 0.5*(q0-2)^2/1 + 0.5*(q1-3)^2/4 so we have exp(-V(q)) as potential energy for the Hamiltonian
#We define the Hamiltonian as H(p,q) = V(q) + K(p) with K(p) = p^2/2 (exp(-K(p)) = exp(-0.5*(p-0)^2/1))
#This means that we can draw momentum variables p from a standard normal distribution N(0,1).

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
import HMC_Stepper_TestCase as HMC

#LEAPFROG METHOD: choose steplength epsilon
epsilon = 1.0e-1
#LEAPFROG METHOD: choose number of steps L
L = 3
#choose initial condition
q = scipy.array([0.0,0.0])
#choose number of samples N
N = int(1e6)

#HMC METHOD
#initialize micro state
print("starting HMC method...")
q0s = scipy.zeros(N) #initial eta is not saved in q0s
q1s = scipy.zeros(N) #initial x_a is not saved in q1s

for n in range(0,N): #exectue loop N times
    q = HMC.HMC_Stepper(epsilon,L,q)
    q0s[n] = q[0]
    q1s[n] = q[1]
print("computations done!")

#the correct distribution
xminq0 = -2.0
xmaxq0 = 6.0
xminq1 = -4.0
xmaxq1 = 10.0
x_coordinate_q0 = scipy.linspace(xminq0,xmaxq0,1000)
x_coordinate_q1 = scipy.linspace(xminq1,xmaxq1,1000)
y_coordinate_q0 = scipy.zeros(scipy.size(x_coordinate_q0))
y_coordinate_q1 = scipy.zeros(scipy.size(x_coordinate_q1))
for index in range(0,len(x_coordinate_q0)):
    y_coordinate_q0[index] = 1.0/(1.0*math.sqrt(2.0*math.pi))*math.exp(-0.5*((x_coordinate_q0[index]-2.0)/1.0)**2)
    y_coordinate_q1[index] = 1.0/(2.0*math.sqrt(2.0*math.pi))*math.exp(-0.5*((x_coordinate_q1[index]-3.0)/2.0)**2)



plt.figure()
plt.hist(q0s,bins=1000,range=[xminq0,xmaxq0],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_q0,y_coordinate_q0)
plt.xlabel('q(1)')
plt.show()

plt.figure()
plt.hist(q1s,bins=1000,range=[xminq1,xmaxq1],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_q1,y_coordinate_q1)
plt.xlabel('q(2)')
plt.show()






