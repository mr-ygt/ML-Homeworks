# Emre YİĞİT 150116056
import numpy as np
import math

e = math.e

lrate = 0.1     # learning rate

# the points
u = 1
v = 1

# this loop to find number of iteration in question 5
for iter in range(1, 100):
    # following lines are for finding partial derivative and calculating the gradient
    dE_du = 2*(u*e**v - 2*v*e**(-u)) * (e**v + 2*v*e**(-u))
    dE_dv = 2*(u*e**v - 2*v*e**(-u)) * (u*e**v - 2*e**(-u))
    gradient = np.array([dE_du, dE_dv])

    # update new points
    u = u - lrate * gradient[0]
    v = v - lrate * gradient[1]

    if ((u*e**v - 2*v*e**(-u))**2) < (10**(-14)):   # check whether the error is less than the given tolerance
        break

print("Number of iterations = ", iter)
u_v = [u, v]
# coordinates of points in answers
points = [(1.000, 1.000), (0.713, 0.045), (0.016, 0.112), (-0.083, 0.029), (0.045, 0.024)]
minDist = 9999

# this loop to find the closest point in question 6
for i in points:
    np_point = np.array(i)
    dist = np.linalg.norm(u_v - np_point)   # distance between the points that
    if dist < minDist:
        minDist = dist
        thePoint = np_point

print("The closest point = [%5.3f, %5.3f] and the distance = %f" % (thePoint[0], thePoint[1], minDist))