#Algorithm that allows to take one step using the HMC method
#V = potential energy
#gradV = gradient of potential energy
#epsilon = steplength in Leapfrog method
#L = number of steps in Leapfrog method
#q_current = current microscopic state

import scipy
import math
import HMC_userDefinedExpressions as UDE
def HMC_Stepper(epsilon, L, q_current):
    q = q_current
    p = scipy.random.normal(0.0, 1.0, scipy.size(q_current)) #choose fictitious momentum variables that are normally distributed
    p_current = p
    #START LEAPFROG METHOD
    #make half step for momentum at the beginning of the Leapfrog method
    p = p - epsilon*UDE.gradV(q)/2.0
    #alternate full steps for microscopic state and momentum
    for l in range (1,L): #We take L-1 steps, last step is done explicitly below the loop
        #make full step for the microscopic state
        q = q + epsilon*p
        #make full step for the momentum
        p = p - epsilon * UDE.gradV(q)
    #final (L-th) step: full step for q, half step for p
    q = q+epsilon*p
    p = p-epsilon * UDE.gradV(q)/2.0
    #END LEAPFROG METHOD

    #negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    #evaluate potential and kinetic energies at start and end of trajectory
    currentV = UDE.V(q_current)
    currentK = scipy.inner(p_current,p_current)/2.0
    proposedV = UDE.V(q)
    proposedK = scipy.inner(p,p)/2.0

    #Accept or reject the state at end of trajectory, returning either
    #the position at the end of the trajectory or the initial position
    U = math.log(scipy.random.uniform(0.0, 1.0))  # generate uniform distributed random number and take logarithm
    testvalue = currentV-proposedV+currentK-proposedK
    if U > testvalue:
        #print("reject")
        q = q_current
    return q