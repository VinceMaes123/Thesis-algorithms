#This file contains all distributions, potentials, gradients... that the user of the HMC method should provide
import scipy
from scipy import linalg
import math

#x_micro = q in the HMC method!!!

########################################################################################################################
################################################# SIMPLE TEST CASE #####################################################

#Simple testcase comprises a 2D normal distribution with expected values 2,3 and variances 1,4
def V(q):
    return 0.5*((1.0/1.0)*(q[0]-2.0)**2 + (1.0/4.0)*(q[1]-3.0)**2)

def gradV(q):
    dV = scipy.zeros(2)
    dV[0] = 0.5*(2.0*(q[0]-2.0))
    dV[1] = 0.5*(2.0*(1.0/4.0)*(q[1]-3.0))
    return dV

