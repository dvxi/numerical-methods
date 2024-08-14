import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

N = 50
ALPHA = np.random.rand() - 0.5

def f_1(x):
    return 2 * np.sin(x) + np.sin(2 * x) + 2 * np.sin(3 * x) + ALPHA

def f_2(x):
    return 2 * np.sin(x) + np.sin(2 * x) + 2 * np.cos(x) + np.cos(2 * x)

def f_3(x):
    return 2 * np.sin(1.1 * x) + np.sin(2.1 * x) + 2 * np.sin(3.1 * x)

def approx_fun(fun, m_s, m_c, x_pts, x):

    A = B = 0

    for k in range(m_s):
        a_sum = 0
        
        for i in range(2 * N):
            a_sum += fun(x_pts[i]) * np.sin(x_pts[i] * k)

        a_k = 1 / N * a_sum
        print("a_k :" + str(a_k))
        A += a_k * np.sin(x * k)
        
    for k in range(m_c):
        b_sum = 0

        for i in range(2 * N):
            b_sum += fun(x_pts[i]) * np.cos(x_pts[i] * k)

        b_k = 1 / N * b_sum
        if(k == 0):
            b_k /= 4

        print("b_k :" + str(a_k))
        B += b_k * np.cos(x * k)

    return (A + B)

def prepare_fun(M_S, M_C, fun):
    x_points = np.zeros(2 * N)
    y_points = np.zeros(2 * N)

    for i in range(2 * N):
        x_points[i] = np.pi * i / N
        y_points[i] = fun(x_points[i])

    M_S = 5
    M_C = 5

    # plt.plot(x_points, y_points)
    # plt.show()

    y_points_app = np.zeros(2 * N)
    for i in range(2 * N):
        y_points_app[i] = approx_fun(fun, 5, 5, x_points, x_points[i])
    
    return x_points, y_points, y_points_app

x_points, y_points, y_points_app = prepare_fun(5, 5, f_1)

plt.plot(x_points, y_points, label="Original f_1")
plt.plot(x_points, y_points_app, label="Approximated f_1")
plt.legend()
plt.show()

x_points, y_points, y_points_app = prepare_fun(5, 5, f_2)

plt.plot(x_points, y_points, label="Original f_2")
plt.plot(x_points, y_points_app, label="Approximated f_2")
plt.legend()
plt.show()

x_points, y_points, y_points_app = prepare_fun(5, 5, f_3)

plt.plot(x_points, y_points, label="Original f_3")
plt.plot(x_points, y_points_app, label="Approximated f_3")
plt.legend()
plt.show()

