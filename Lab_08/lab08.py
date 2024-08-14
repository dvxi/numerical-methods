import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

N = 5
X_0 = -5
X_1 = 5
DX = 0.01

def f(x):
    return 1 / (1 + x**2)

def f2(x):
    return np.cos(2*x)

def wyzM(xw, yw, n, alpha, beta):
    m = np.zeros([n])
    d = np.zeros([n])
    A = np.zeros([n, n])

    d[0] = alpha
    d[n-1] = beta
    A[0][0] = 1

    for i in range(1, n - 1):
        h_i = xw[i] - xw[i - 1]
        h_i_1 = xw[i + 1] - xw[i]
        lbd_i = h_i_1 / (h_i + h_i_1)
        mi_i = 1 - lbd_i
        d_i = (6.0 / (h_i + h_i_1)) * (((yw[i+1] - yw[i]) / h_i_1) - ((yw[i] - yw[i-1]) / h_i))

        A[i][i] = 2
        A[i][i - 1] = mi_i
        A[i][i + 1] = lbd_i

        d[i] = d_i

    A[n - 1][n - 1] = 1

    A_inv = inv(A)
    m = np.matmul(A_inv, d)

    return m

def wyzSx(xw, yw, m, n, x):
    i = 0
    while(xw[i] < x and i < n):
        i += 1

    h_i = xw[i] - xw[i - 1] # Co gdy w pierwszym przedziale?

    A_i = ((yw[i] - yw[i - 1]) / (h_i)) - (h_i / 6.0) * (m[i] - m[i - 1])
    B_i = yw[i - 1] - m[i - 1] * (pow(h_i, 2) / 6.0)

    sx = m[i - 1] * (pow(xw[i] - x, 3) / (6.0 * h_i)) + m[i] * (pow(x - xw[i - 1], 3) / (6.0 * h_i)) + A_i * (x - xw[i - 1]) + B_i

    return sx

def wyzApproxSecondDerivative(f, x, delta_x):
    return (f(x - delta_x) - 2*f(x) + f(x + delta_x)) / (delta_x**2)

dx = 10 / (N - 1)

xw = np.linspace(-5, 5, N)
yw = f(xw)

print("xw:")
print(xw)

m = wyzM(xw, yw, N, 0, 0)
sx = wyzSx(xw, yw, m, N, -1.0)

print("m:")
print(m)

print("sx:")
print(sx)

x_new = np.linspace(-5, 5, 100)
y_new = f(x_new)

i = 0
for iks in x_new:
    y_new[i] = wyzSx(xw, yw, m, 100, iks)

plt.plot(x_new, y_new)
plt.show()
plt.plot(x_new, f(x_new))
plt.show()

# Drugie pochodne
delta_x = 0.01
approx_second_derivatives = [wyzApproxSecondDerivative(f, x, delta_x) for x in xw]
exact_second_derivatives = [f2(x) for x in xw]

plt.plot(xw, approx_second_derivatives, label='Aproksymowane')
plt.plot(xw, exact_second_derivatives, label='Dokładne')
plt.xlabel('Węzły interpolacji')
plt.ylabel('Drugie pochodne')
plt.legend()
plt.title('Porównanie aproksymowanych i dokładnych drugich pochodnych')
plt.show()

# Interpolacja dla f1(x) oraz f2(x) w przedziale x ∈ [−5,5], dla liczby węzłów: n = 5, 8, 21
n_values = [5, 8, 21]
for n in n_values:
    xw = np.linspace(-5, 5, n)
    yw = f(xw)
    m = wyzM(xw, yw, n, 0, 0)
    x_new = np.linspace(-5, 5, 100)
    y_new_interpolated = [wyzSx(xw, yw, m, n, x) for x in x_new]

    plt.plot(x_new, y_new_interpolated, label=f'n = {n}')
plt.plot(x_new, f(x_new), label='f(x)')
plt.legend()
plt.title('Interpolacja dla f1(x)')
plt.show()

for n in n_values:
    xw = np.linspace(-5, 5, n)
    yw = f2(xw)
    m = wyzM(xw, yw, n, 0, 0)
    x_new = np.linspace(-5, 5, 100)
    y_new_interpolated = [wyzSx(xw, yw, m, n, x) for x in x_new]

    plt.plot(x_new, y_new_interpolated, label=f'n = {n}')
plt.plot(x_new, f2(x_new), label='f2(x)')
plt.legend()
plt.title('Interpolacja dla f2(x)')
plt.show()
