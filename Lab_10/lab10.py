import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

DELTA = pow(10, -4)
H = 0.1

def f_xy(x, y):
    return (5/2) * pow((pow(x, 2) - y), 2) + pow(1 - x, 2)

def grad_f(f, x, y):
    f_x = (f(x + DELTA, y) - f(x - DELTA, y)) / (2 * DELTA)
    f_y = (f(x, y + DELTA) - f(x, y - DELTA)) / (2 * DELTA)

    return np.array([f_x, f_y])

def vec_dist(r_0, r_1):
    return np.sqrt(pow(r_0[0] - r_1[0], 2) + pow(r_0[1] - r_1[1], 2))

def find_min(f, x_0, y_0, eps):

    if(eps == 0.01):
        file = open("eps1.dat", 'w')
        file.write("--- Epsilon = 10^-2 ---\n")
        eps01_path.fill(0)
    else:
        file = open("eps2.dat", "w")
        file.write("--- Epsilon = 10^-3 ---\n")
        eps01_path.fill(0)

    r_i = np.array([x_0, y_0])
    for i in range(1000):
        if(eps == 0.01):
            eps01_path[i] = r_i
        else:
            eps001_path[i] = r_i

        r_i_1 = r_i - H * grad_f(f, r_i[0], r_i[1])
        print("x: " + str(r_i[0]) + ", y: " + str(r_i[1]))

        file.write("x: " + str(r_i[0]) + ", y: " + str(r_i[1]) + "\n")

        if(vec_dist(r_i, r_i_1) < eps):
            print("Spelnione po " + str(i + 1) + " iteracjach.")
        r_i = r_i_1
    
    file.close()

print("--- Epsilon = 10^-2 ---")
eps01_path = np.zeros((1000, 2))
find_min(f_xy, -0.75, 1.75, 0.01)

# print("Path of 0.01 path:")
# for row in eps01_path:
#     print(row)

print("--- Epsilon = 10^-3 ---")
eps001_path = np.zeros((1000, 2))
find_min(f_xy, -0.75, 1.75, 0.001)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f_xy(X, Y)

plt.contour(X, Y, Z, levels=100, extent=[-2, 2, -2, 2])
plt.colorbar()
plt.plot(eps001_path[:, 0], eps001_path[:, 1], 'r-', label='eps: 0.001 path')
plt.legend()
plt.plot(eps01_path[:, 0], eps01_path[:, 1], 'b-', label='eps: 0.01 path')
plt.legend()
plt.show()