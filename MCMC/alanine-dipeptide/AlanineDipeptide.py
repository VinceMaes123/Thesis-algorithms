#Main script: perform N steps with the mM-MCMC method and plot the distributions
import scipy
from scipy import integrate
from scipy import io
import math
import settings
import matplotlib.pyplot as plt
import userDefinedExpressionsAlanineDipeptide as UDE
import stepperAlanineDipeptide as stepper
import acceptRejectAlanineDipeptide as AR
import precomputationsAlanineDipeptide as prec

#initialize solver object and perform precomputations
print("initializing...")
eps = 1.0e-6
factor = 1.0
N = int(1e5)
K = 8
macroTimestep = 0.001
beta = 1.0/100.0
solver = settings.Solver(eps=eps,landa=factor/eps,microTimestep=eps/factor,N=N,K=K,macroTimestep=macroTimestep,beta=beta,zmin=-math.pi,zmax=math.pi)
z_grid = prec.getMacroGrid(solver)  # grid at macro level used for precomputations
A_tilde_grid = prec.getA_tilde(z_grid,solver)  #precomputed A_tilde in z_grid
print("done initializing!")

#initialize micro state
print("starting mM-MCMC method...")
x_micro = scipy.array([1.53,1.53,1.335,1.335,1.335,1.335,(113.9 / 360.0) * 2.0 * math.pi,(113.9 / 360.0) * 2.0 * math.pi,
                 (113.9 / 360.0) * 2.0 * math.pi,(117.6 / 360.0) * 2.0 * math.pi,(117.6 / 360.0) * 2.0 * math.pi,
                 0.0,0.0]) #these are all equilibrium values
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
    CC1[n] = x_micro[0]
    CC2[n] = x_micro[1]
    CN1[n] = x_micro[2]
    CN2[n] = x_micro[3]
    CN3[n] = x_micro[4]
    CN4[n] = x_micro[5]
    CCN1[n] = x_micro[6]
    CCN2[n] = x_micro[7]
    CCN3[n] = x_micro[8]
    CNC1[n] = x_micro[9]
    CNC2[n] = x_micro[10]
    phi[n] = x_micro[11]
    psi[n] = x_micro[12]
print("computations done!")

#the correct distribution
xminPsi = -1.05
xmaxPsi = 1.05
scaling_Psi = integrate.quad(UDE.mu_0,xminPsi,xmaxPsi,args=(solver,))
scaling_Psi = scaling_Psi[0]
x_coordinate_Psi = scipy.linspace(xminPsi,xmaxPsi,1000)
y_coordinate_Psi = scipy.zeros(scipy.size(x_coordinate_Psi))
for index in range(0,len(x_coordinate_Psi)):
    y_coordinate_Psi[index] = (1.0/scaling_Psi)*UDE.mu_0(x_coordinate_Psi[index],solver)



plt.figure()
plt.hist(psi,bins=1000,range=[xminPsi,xmaxPsi],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_Psi,y_coordinate_Psi)
plt.xlabel(r'Torsion Angle $\psi$')
plt.show()


UDE.makePlot(phi,-0.25,0.25,UDE.phi_0,solver,r'Torsion angle $\phi$')
UDE.makePlot(CC1,1.515-0.05,1.515+0.05,UDE.CC_0,solver,r'Bond length CC')
UDE.makePlot(CN1,1.335-0.05,1.335+0.05,UDE.CN_0,solver,r'Bond length CN')
UDE.makePlot(CCN1,(113.9 / 360.0) * 2.0 * math.pi-0.10,(113.9 / 360.0) * 2.0 * math.pi+0.10,UDE.CCN_0,solver,r'Bond angle CCN')
UDE.makePlot(CNC1,(117.6 / 360.0) * 2.0 * math.pi-0.10,(117.6 / 360.0) * 2.0 * math.pi+0.10,UDE.CNC_0,solver,r'Bond angle CNC')




