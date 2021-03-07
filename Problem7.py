# Emre YİĞİT 150116056
import numpy as np
import math

e = math.e

lrate = 0.1     # learning rate

# the points
u = 1
v = 1
# this loop to find the error
for iter in range(15):
    # following lines are for finding partial derivative, calculating the gradient and update the points
    dE_du = 2*(u*e**v - 2*v*e**(-u)) * (e**v + 2*v*e**(-u))
    gradient = np.array([dE_du, 0])

    u = u - lrate * gradient[0]

    dE_dv = 2*(u*e**v - 2*v*e**(-u)) * (u*e**v - 2*e**(-u))
    gradient = np.array([0, dE_dv])
    v = v - lrate * gradient[1]

error = (u*e**v - 2*v*e**(-u))**2

print("error = ", error)