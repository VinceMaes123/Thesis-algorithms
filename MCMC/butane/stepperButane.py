#this file contains all steppers used in the classic MCMC method and mM-MCMC method
import scipy
from scipy import linalg
import math
import userDefinedExpressionsButane as UDE

def classicMCMC(x_current,solver):
    # generate a step according to dX = -grad(V)dt+sqrt(2)dW
    gradV = UDE.totalgradPot(x_current,solver)
    dW = math.sqrt(solver.microTimestep) * scipy.random.normal(0.0, 1.0, len(x_current))  # brownian step at micro level
    x_new = x_current - gradV * solver.microTimestep + math.sqrt(2.0/solver.beta)*dW
    return x_new

def indirect(x_current,eta,solver):
    # generate a step according to dX = -grad(V+landa/2*||eta(x_current)-eta_coarseStep||^2)dt+sqrt(2)dW
    gradV = UDE.totalgradPot(x_current,solver)
    z_current = UDE.restriction(x_current)
    gradE = UDE.gradRestriction(x_current)
    dW = math.sqrt(solver.microTimestep) * scipy.random.normal(0.0, 1.0, len(x_current)) #brownian step at micro level
    x_new = x_current - gradV * solver.microTimestep - \
            solver.landa * (z_current - eta) * gradE * solver.microTimestep + math.sqrt(2.0 / solver.beta) * dW
    return x_new


def coarse(eta_current,solver):
    #generate a step according to dEta = b*dt+sqrt(2*beta^-1)*sigma*dW
    dW = math.sqrt(solver.macroTimestep) * scipy.random.normal(0.0, 1.0) #brownian step according to taken timestep on the macro level
    deta = UDE.b_exact(eta_current) * solver.macroTimestep + \
           math.sqrt(2.0 / solver.beta) * UDE.sigma_exact(eta_current) * dW #proposed displacement
    eta_new = eta_current+deta
    return eta_new

