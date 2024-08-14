import numpy as np
import numpy.polynomial as polynomial
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)

# 1.

def f(x):
    return x / (4 * np.power(x, 2) + 1)

def F(x, a = 2, c = 1):
    return 1 / (2 * pow(a, 2)) * np.log(pow(a, 2) * np.power(x, 2) + c)

def t(x, a, b):
    return (a + b) / 2 + (b - a) / 2 * x

def g(x, A = 0, B = 2):
    return f(t(x, A, B))

A = 0
B = 2
INTEGRAL = F(B) - F(A)

print(f'INTEGRAL = {INTEGRAL}')

differencesTable = []

for N in range(2, 21):
    x, w = polynomial.legendre.leggauss(N)
    integral = 0
    for i in range(N):
        integral += w[i] * g(x[i])
    integral *= (B - A) / 2
    print(f'N = {N}, integral = {integral}')
    differencesTable.append(abs(INTEGRAL - integral))

plt.plot(range(2, 21), differencesTable)
plt.title('Differences between real and calculated integrals c1')
plt.show()

# 2.

def f_2(x, k):
    return np.power(x, k) #* np.exp(-x)

def F_2(k, x = 0):
    return math.factorial(k)

def g_2(x, k):
    return f_2(t(x, 0, 1), k)

kTable = [5, 10]

for k in kTable:
    print(f'k = {k}')
    INTEGRAL = F_2(k)
    differencesTable = []

    for N in range(2, 21):
        x, w = polynomial.laguerre.laggauss(N)
        # print(f'w = {w}')
        integral = 0
        for i in range(N):
            integral += w[i] * f_2(x[i], k)
        # print(f'N = {N}, integral = {integral}')
        differencesTable.append(abs(INTEGRAL - integral))

    plt.plot(range(2, 21), differencesTable)
    plt.title('Differences between real and calculated integrals c2 k = ' + str(k))
    plt.show()

# 3.

C_dok = 0.1919832644

def f_3_x(x):
    return np.power(np.sin(x), 2) #* np.exp(-np.power(x, 2))

def f_3_y(y):
    return np.power(np.sin(y), 4) #* np.exp(-np.power(y, 2))

differencesTable = []

for N in range(2, 16):
    x, w = polynomial.hermite.hermgauss(N)
    integral_1 = 0
    integral_2 = 0

    for i in range(N):
        integral_1 += w[i] * f_3_x(x[i])
    for j in range(N):
        integral_2 += w[j] * f_3_y(x[j])

    integral = integral_1 * integral_2
    differencesTable.append(abs(C_dok - integral))
    # print(f'N = {N}, integral = {integral}')
    # print(f'N = {N}, difference = {abs(C_dok - integral)}')

plt.plot(range(2, 16), differencesTable)
plt.title('Differences between real and calculated integrals c3')
plt.show()