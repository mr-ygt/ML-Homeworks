# Emre YİĞİT 150116056
import random
import numpy as np
import math

l_rate = 0.01   # learning rate
e = math.e
total_epoch = 0
total_E_out = 0

for run in range(100):      # repeat the experiment for 100 runs

    # creating points of line
    line_point1 = np.random.uniform(-1, 1, 2)
    line_point2 = np.random.uniform(-1, 1, 2)
    # equation of line y = m*x + b
    slope = ((line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0]))  # y-y1 / x-x1
    intercept = line_point2[1] - slope * line_point2[0]  # b = y - m*x
    weight_fx = np.array([intercept, slope, -1])    # weight for f(x)
    # creating points which in [-1, 1] x [-1, 1]
    point1 = np.random.uniform(-1, 1, 100)  # N = 100
    point2 = np.random.uniform(-1, 1, 100)
    all_points = np.array([np.ones(100), point1, point2])
    all_points = all_points.T  # Transpose the array to match weight_fx
    # this array keeps the classification of point based on line ( label them -1 or +1)
    point_location = np.sign(np.dot(all_points, weight_fx))

    weight_gx = np.zeros(3)  # weight for g(x)
    weight_gx_1 = np.ones(3)  # w_t+1
    epoch = -1
    # stop the algorithm when (w_t-1 - w_t) < 0.01
    while np.linalg.norm(weight_gx - weight_gx_1) > 0.01:
        epoch = epoch + 1
        indices = list(range(100))
        random.shuffle(indices)     # the array with random numbers from 0 to 99
        weight_gx_1 = weight_gx     # weight_t-1 = weight_t
        # following loop picking randomly a point and and calculating new weight
        for j in indices:
            x_n = all_points[j, :]
            y_n = point_location[j]
            gradient = ((-y_n * x_n) / (e ** (y_n * np.dot(weight_gx.T, x_n)) + 1))

            weight_gx = weight_gx - l_rate * gradient       # update new weight

    total_epoch = total_epoch + epoch

    #   *********************   answering questions   *********************
    # creating different 100 points to calculate E_out
    point2_1 = np.random.uniform(-1, 1, 100)
    point2_2 = np.random.uniform(-1, 1, 100)
    all_points2 = np.array([np.ones(100), point2_1, point2_2])
    all_points2 = all_points2.T
    point_location2 = np.sign(np.dot(all_points2, weight_fx))
    E_out = 0
    for iter2 in range(100):
        E_out = E_out + np.log(e ** (-point_location2[iter2] * np.dot(all_points2[iter2, :], weight_gx)) + 1)
    total_E_out = total_E_out + E_out/100

avg_E_out = total_E_out/100
avg_epoch = total_epoch/100

print("Average E_out = ", avg_E_out, " Average epoch = ", avg_epoch)