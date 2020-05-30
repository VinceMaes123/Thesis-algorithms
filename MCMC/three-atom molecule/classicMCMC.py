#Implementation of the standard MCMC method applied to the 3-atom molecule example.
import scipy
from scipy import integrate
from scipy import io
import math
import settings
import matplotlib.pyplot as plt
import userDefinedExpressions as UDE
import stepper
import acceptReject as AR

solver = settings.Solver(x_init=[1.0,math.cos(math.pi/2.0-0.3838),math.sin(math.pi/2.0-0.3838)],N=int(1e6)) #default settings

#initialize microscopic sample
print("starting classic MCMC method...")
scipy.random.seed(256) #always generate the same figures!
x_micro = solver.x_init
eta = scipy.zeros(solver.N) #initial eta is not saved in eta
xas = scipy.zeros(solver.N) #initial x_a is not saved in xas
rcs = scipy.zeros(solver.N) #initial r_c is not saved in rcs
xcs = scipy.zeros(solver.N) #initial r_c is not saved in rcs
ycs = scipy.zeros(solver.N) #initial r_c is not saved in rcs

for n in range(0,solver.N): #exectue loop N times
    #perform standard MCMC
    x_new = stepper.classicMCMC(x_micro,solver)
    x_new = AR.classic(x_micro,x_new,solver)
    x_micro = x_new
    #save variables in order to plot them below
    eta[n] = UDE.restriction(x_micro)
    xas[n] = x_micro[0]
    rcs[n] = math.sqrt(x_micro[1]**2 + x_micro[2]**2)
    xcs[n] = x_micro[1]
    ycs[n] = x_micro[2]
print("computations done!")

#the correct densities
scaling_eta = integrate.quad(UDE.mu_0,0.0,3.0)
scaling_eta = scaling_eta[0]
x_coordinate_eta = scipy.linspace(0.0,3.0,1000)
y_coordinate_eta = scipy.zeros(scipy.size(x_coordinate_eta))
for index in range(0,len(x_coordinate_eta)):
    y_coordinate_eta[index] = (1.0/scaling_eta)*UDE.mu_0(x_coordinate_eta[index])

scaling_c = integrate.quad(UDE.distplot,0.9,1.1,args=(solver.eps))
scaling_c = scaling_c[0]
x_coordinate_c = scipy.linspace(0.99,1.01,1000)
y_coordinate_c = scipy.zeros(scipy.size(x_coordinate_c))
for index in range(0,len(x_coordinate_c)):
    y_coordinate_c[index] = (1.0/scaling_c)*math.exp((-0.5)*(x_coordinate_c[index]-1.00)**2/solver.eps)


plt.figure()
plt.hist(eta,bins=1000,range=[0.0,3.0],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_eta,y_coordinate_eta)
#plt.title("eta")
plt.xlabel(r'$\theta$')
plt.show()

plt.figure()
plt.hist(xas,bins=1000,range=[0.990,1.010],density=1)
plt.plot(x_coordinate_c,y_coordinate_c)
plt.title("x_a")
plt.show()

plt.figure()
plt.hist(xcs,bins=1000,density=1)
plt.title("x_c")
plt.show()

plt.figure()
plt.hist(ycs,bins=1000,density=1)
plt.title("y_c")
plt.show()

plt.figure()
plt.hist(rcs,bins=1000,range=[0.990,1.010],density=1)
plt.plot(x_coordinate_c,y_coordinate_c)
plt.xlabel(r'$r_c$')
#plt.title("r_c")
plt.show()
#plt.savefig('test.pdf') #save data to a pdf
#scipy.io.savemat('classic_MCMC',{'xas':xas,'xcs':xcs,'ycs':ycs}) #save to .mat file that can be interpreted by MATLAB


