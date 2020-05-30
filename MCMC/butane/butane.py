#Main script: perform N steps with the mM-MCMC method and plot the distributions
import scipy
from scipy import integrate
from scipy import io
import math
import settings
import matplotlib.pyplot as plt
import userDefinedExpressionsButane as UDE
import stepperButane as stepper
import acceptRejectButane as AR
import precomputationsButane as prec

#initialize solver object and perform precomputations
print("initializing...")
eps = 1.0e-6
factor = 1.0
N = int(1e5)
K = 5
macroTimestep = 5.0e-4
beta = 1.0/200.0
solver = settings.Solver(eps=eps,landa=factor/eps,microTimestep=eps/factor,N=N,K=K,macroTimestep=macroTimestep,beta=beta,zmin=-math.pi,zmax=math.pi)
z_grid = prec.getMacroGrid(solver)  # grid at macro level used for precomputations
A_tilde_grid = prec.getA_tilde(z_grid,solver)  #precomputed A_tilde in z_grid
print("done initializing!")

#initialize micro state
print("starting mM-MCMC method...")
x_micro = scipy.array([1.53,1.53,1.53,(112.0/360.0)*2.0*math.pi,(112.0/360.0)*2.0*math.pi,0.0])
BL1 = scipy.zeros(N) #Bond lengths
BL2 = scipy.zeros(N)
BL3 = scipy.zeros(N)
BA1 = scipy.zeros(N) #Bond angles
BA2 = scipy.zeros(N)
TA = scipy.zeros(N) #Torsion angle

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
    BL1[n] = x_micro[0]
    BL2[n] = x_micro[1]
    BL3[n] = x_micro[2]
    BA1[n] = x_micro[3]
    BA2[n] = x_micro[4]
    TA[n] = x_micro[5]
print("computations done!")

#the correct distribution
xminTA = -math.pi
xmaxTA = math.pi
scaling_TA = integrate.quad(UDE.mu_0,xminTA,xmaxTA,args=(solver,))
scaling_TA = scaling_TA[0]
x_coordinate_TA = scipy.linspace(xminTA,xmaxTA,1000)
y_coordinate_TA = scipy.zeros(scipy.size(x_coordinate_TA))
for index in range(0,len(x_coordinate_TA)):
    y_coordinate_TA[index] = (1.0/scaling_TA)*UDE.mu_0(x_coordinate_TA[index],solver)



plt.figure()
plt.hist(TA,bins=1000,range=[xminTA,xmaxTA],density=1) #histogram with 1000 bins
plt.plot(x_coordinate_TA,y_coordinate_TA)
plt.xlabel('Torsion Angle')
plt.show()

UDE.makePlot(BL1,1.53-0.10,1.53+0.10,UDE.CC_0,solver,r'Bond length CC')
UDE.makePlot(BA1,(112.0 / 360.0) * 2.0 * math.pi-0.20,(112.0 / 360.0) * 2.0 * math.pi+0.20,UDE.CCC_0,solver,r'Bond angle CCC')


