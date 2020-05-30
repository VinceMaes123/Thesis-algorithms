#This file contains all methods concerning precomputations
import scipy
from scipy import linalg
import math
import userDefinedExpressionsAlanineDipeptide as UDE
import stepperAlanineDipeptide as stepper
import acceptRejectAlanineDipeptide as AR

#generate a grid over the macro variables
def getMacroGrid(solver):
    z_grid = scipy.linspace(solver.zmin, solver.zmax, solver.J)
    return z_grid

#calculate A_tilde on z_grid
def getA_tilde(z_grid,solver):
    A_tilde = scipy.zeros(scipy.size(z_grid))
    for index in range(0,scipy.size(z_grid)):
        A_tilde[index] = (1.0 / solver.N_gauss) * UDE.getAtilde(z_grid[index],solver)
    return A_tilde

#interpolate on z_grid (works only for 1D z_grid)
def interpolate(z_value,z_grid,func_grid):
    rightindex = 0
    #find index of z_grid value to the right of given z_value
    while(z_value > z_grid[rightindex]):
        rightindex = rightindex+1
    leftindex = rightindex-1
    func_value = func_grid[leftindex] + \
                 (func_grid[rightindex]-func_grid[leftindex])/(z_grid[rightindex]-z_grid[leftindex]) * (z_value-z_grid[leftindex])
    return func_value

#generate a list of N_prec samples for each gridpoint in z_grid. Based on these samples,
#the macroscopic dynamics b, sigma and the free energy A_hat are calculated on the grid.
#WORKS ONLY IF 1 MACRO VARIABLE
#This version uses normal weights (1/N_prec) for the free energy A_hat
def precomputation(z_grid,solver):
    b_grid = scipy.zeros(scipy.size(z_grid))
    sigma_grid = scipy.zeros(scipy.size(z_grid))
    Ahat_grid = scipy.zeros(scipy.size(z_grid))
    for j in range(0,solver.J):
        b = 0.0
        sigma = 0.0
        Ahat_temp = 0.0
        eta = z_grid[j]
        x_current = solver.x_init
        for n in range(0,solver.N_prec):
            #take a biased step
            x_new = stepper.indirect(x_current,eta,solver)
            #accept-reject
            x_new = AR.indirect(x_current,x_new,eta,solver)
            #calculate contribution step
            b = b + (1.0/solver.N_prec)*(-scipy.inner(UDE.totalgradPot(x_new,solver),UDE.gradRestriction(x_new)) +\
                (1.0/solver.beta)*UDE.lapRestriction(x_new))
            sigma = sigma + (1.0/solver.N_prec)*(linalg.norm(UDE.gradRestriction(x_new)))**2
            Ahat_temp = Ahat_temp + (1.0/solver.N_prec)*(math.exp(-solver.beta*UDE.totalPot(x_new,solver)) * \
                        (scipy.inner(UDE.gradRestriction(x_new),UDE.gradRestriction(x_new)))**(-0.5))
            #reset x_current for next iteration in n-loop
            x_current = x_new
        #save contribution in right register
        b_grid[j] = b
        sigma_grid[j] = sigma
        Ahat_grid[j] = -(1.0/(solver.beta))*math.log(Ahat_temp)
    return b_grid,sigma_grid,Ahat_grid

#Alternative way to get free energy A_hat that only works for overdamped Langeving dynamics where b = -grad(A_hat)
#This method integrates "-b" to get A_hat
def get_A_hat_Alternative(z_grid,b_grid):
    A_hat = scipy.zeros(scipy.size(z_grid))
    dz = z_grid[1] - z_grid[0]

    for index in range(1, scipy.size(z_grid)):
        A_hat[index] = A_hat[index - 1] - dz * b_grid[index - 1]

    A_hat = A_hat - min(A_hat)
    return A_hat