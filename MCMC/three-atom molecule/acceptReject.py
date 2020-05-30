#This file contains all functions concerning accept-reject steps
import math
import scipy
from scipy import linalg
import userDefinedExpressions as UDE
import precomputations as prec

#standard MCMC
#argument exponential invariant density
def classic_arg(x_micro,solver):
    arg = -solver.beta * UDE.totalPot(x_micro,solver)
    return arg

#argument exponential transition density
def classic_bar_arg(x_current,x_new,solver):
    gradV = UDE.totalgradPot(x_current,solver)
    arg = -linalg.norm(x_new - x_current + gradV * solver.microTimestep) ** 2
    arg = arg * solver.beta / (4.0 * solver.microTimestep)
    return arg

#Accept-reject for standard MCMC
def classic(x_current,x_new,solver):
    U = math.log(scipy.random.uniform(0.0, 1.0))  # generate uniform distributed random number and take logarithm
    testvalue = classic_arg(x_new,solver)-classic_arg(x_current,solver) + \
                classic_bar_arg(x_new,x_current,solver)-classic_bar_arg(x_current,x_new,solver)
#    print(math.exp(testvalue))
    if U > testvalue:
#        print("reject")
        x_new = x_current  # reject the step
    return x_new


########################################################################################################################
#Indirect reconstruction

#argument indirect reconstruction density
def nu_arg(x_micro,eta,solver):
    arg = -solver.beta * (UDE.totalPot(x_micro,solver) + (solver.landa / 2.0) * (linalg.norm(UDE.restriction(x_micro) - eta)) ** 2)
    return arg

#argument exponential transition density biased dynamics
def nu_bar_arg(x_current,x_new,eta,solver):
    gradV = UDE.totalgradPot(x_current,solver)
    z_current = UDE.restriction(x_current)
    gradE = UDE.gradRestriction(x_current)
    arg = -linalg.norm(x_new - x_current + gradV * solver.microTimestep +
                       solver.landa * (z_current - eta) * gradE * solver.microTimestep) ** 2
    arg = arg * solver.beta / (4.0 * solver.microTimestep)
    return arg

#Accept-reject for the indirect reconstruction step
def indirect(x_current,x_new,eta,solver):
    U = math.log(scipy.random.uniform(0.0, 1.0))  # generate uniform distributed random number and take logarithm
    testvalue = nu_arg(x_new,eta,solver)-nu_arg(x_current,eta,solver) + \
                nu_bar_arg(x_new,x_current,eta,solver)-nu_bar_arg(x_current,x_new,eta,solver)
#    print(math.exp(testvalue))
    if U > testvalue:
#        print("reject")
        x_new = x_current  # reject the step
    return x_new


########################################################################################################################
#Reaction coordinate

#argument exponential transition density effective dynamics (reaction coordinate)
def q0_arg(eta_current,eta_new,solver):
    arg = -(solver.beta) * (eta_new - eta_current - UDE.b_exact(eta_current) * solver.macroTimestep) ** 2 / (4.0 * solver.macroTimestep)
    return arg

#Accept-reject for the reaction coordinate
def coarse(eta_current,eta_new,solver):
    U = math.log(scipy.random.uniform(0.0, 1.0))  # generate uniform distributed random number and take logarithm
    testvalue = -solver.beta*UDE.A_hat(eta_new) + solver.beta*UDE.A_hat(eta_current) + \
                q0_arg(eta_new,eta_current,solver)-q0_arg(eta_current,eta_new,solver)
    if U > testvalue:
        eta_new = eta_current  # reject the step
    return eta_new

########################################################################################################################
#final proposal new microscopic sample

#Accept-reject for the final proposal of the new microscopic sample
def proposal(x_current,z_current,x_new,z_new,z_grid,A_tilde_grid,solver):
    U = math.log(scipy.random.uniform(0.0, 1.0))  # generate uniform distributed random number and take logarithm
    testarg1 = solver.beta * (-UDE.A_hat(z_current) + UDE.A_hat(z_new))
    testarg2 = math.log(prec.interpolate(z_new,z_grid,A_tilde_grid) /
                        prec.interpolate(z_current,z_grid,A_tilde_grid))
    testvalue = testarg1+testarg2
    if U > testvalue:
        x_new = x_current  # reject the step
        z_new = z_current
        #print("Final proposal rejected!")
    return x_new, z_new
