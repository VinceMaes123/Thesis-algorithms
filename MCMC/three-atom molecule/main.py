#Main script: perform N steps with the mM-MCMC method and plot the distributions
import scipy
from scipy import integrate
from scipy import io
import math
import settings
import matplotlib.pyplot as plt
import userDefinedExpressions as UDE
import stepper
import acceptReject as AR
import precomputations as prec

#initialize solver object and perform precomputations
print("initializing...")
eps = 1.0e-6
factor = 1.0
N = int(1e5)
K = 5
solver = settings.Solver(eps=eps,landa=factor/eps,microTimestep=eps/factor,N=N,K=K)
z_grid = prec.getMacroGrid(solver)  # grid at macro level used for precomputations
A_tilde_grid = prec.getA_tilde(z_grid,solver)  #precomputed A_tilde in z_grid
print("done initializing!")

#initialize micro state
print("starting mM-MCMC method...")
x_micro = solver.x_init
eta = scipy.zeros(solver.N) #initial eta is not saved in eta
xas = scipy.zeros(solver.N) #initial x_a is not saved in xas
rcs = scipy.zeros(solver.N) #initial r_c is not saved in rcs
xcs = scipy.zeros(solver.N) #initial r_c is not saved in rcs
ycs = scipy.zeros(solver.N) #initial r_c is not saved in rcs

#initialize reaction coordinate
eta_current = UDE.restriction(x_micro)

for n in range(0,solver.N): #execute loop N times
    #perform coarse step
    eta_new = stepper.coarse(eta_current,solver)
    #accept-reject coarse step
    eta_new = AR.coarse(eta_current,eta_new,solver)
    if eta_new != eta_current: #only reconstruction if step is accepted
        #reconstruction
        x_current = x_micro
        for k in range(0,solver.K):
            #take biased step
            x_new = stepper.indirect(x_current,eta_new,solver)
            #accept-reject biased step
            x_new = AR.indirect(x_current,x_new,eta_new,solver)
            x_current = x_new
        #accept-reject the final proposal for the new microstate
        x_micro,eta_current = AR.proposal(x_micro,eta_current,x_new,eta_new,z_grid,A_tilde_grid,solver)
    eta[n] = UDE.restriction(x_micro) #eta_new
    xas[n] = x_micro[0]
    rcs[n] = math.sqrt(x_micro[1]**2 + x_micro[2]**2)
    xcs[n] = x_micro[1]
    ycs[n] = x_micro[2]
print("computations done!")

#the correct distribution
scaling_eta = integrate.quad(UDE.mu_0,0.0,3.0)
scaling_eta = scaling_eta[0]
x_coordinate_eta = scipy.linspace(0.0,3.0,1000)
y_coordinate_eta = scipy.zeros(scipy.size(x_coordinate_eta))
for index in range(0,len(x_coordinate_eta)):
    y_coordinate_eta[index] = (1.0/scaling_eta)*UDE.mu_0(x_coordinate_eta[index])

scaling_c = integrate.quad(UDE.distplot,0.8,1.2,args=(solver.eps))
scaling_c = scaling_c[0]
x_coordinate_c = scipy.linspace(0.8,1.2,1000)
y_coordinate_c = scipy.zeros(scipy.size(x_coordinate_c))
for index in range(0,len(x_coordinate_c)):
    y_coordinate_c[index] = (1.0/scaling_c)*math.exp((-0.5)*(x_coordinate_c[index]-1.00)**2/solver.eps)

xmax = 1.0+math.sqrt((solver.eps)/(1.0e-2)) #These limits yield nice figures for the xas and rcs distributions
xmin = 1.0-math.sqrt((solver.eps)/(1.0e-2)) #Limits scale with sqrt(eps) because that is the std of the distribution

plt.figure()
plt.hist(eta,bins=1000,range=[0.0,3.0],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_eta,y_coordinate_eta)
#plt.title("eta")
plt.xlabel(r'$\theta$')
plt.show()

plt.figure()
plt.hist(xas,bins=1000,range=[xmin,xmax],density=1)
plt.plot(x_coordinate_c,y_coordinate_c)
plt.xlim((xmin,xmax))
#plt.title("x_a")
plt.xlabel(r'$x_a$')
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
plt.hist(rcs,bins=1000,range=[xmin,xmax],density=1)
plt.plot(x_coordinate_c,y_coordinate_c)
plt.xlim((xmin,xmax))
#plt.title("r_c")
plt.xlabel(r'$r_c$')
plt.show()
#plt.savefig('test.pdf') #save data to a pdf
#scipy.io.savemat('mM_MCMC',{'xas':xas,'xcs':xcs,'ycs':ycs}) #save to .mat file that can be interpreted by MATLAB


