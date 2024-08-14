import numpy as np
import matplotlib.pyplot as plt

N = 65
X_MIN = -5
X_MAX = 5

def f(x):
    return np.exp(-x**2)

def lagrange_basis(x, i, nodes):
    L = 1
    for j in range(len(nodes)):
        if j != i:
            L *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return L

def lagrange_interpolation(x, nodes, values):
    P = 0
    for i in range(len(values)):
        P += values[i] * lagrange_basis(x, i, nodes)
    return P

def plot_interpolation(n):
    x_min, x_max = -5, 5
    x_nodes = np.linspace(x_min, x_max, n+1)
    y_nodes = f(x_nodes)
    
    x_dense = np.linspace(x_min, x_max, 400)
    y_dense = [lagrange_interpolation(x, x_nodes, y_nodes) for x in x_dense]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, f(x_dense), label='Original Function $f(x) = e^{-x^2}$')
    plt.plot(x_dense, y_dense, label=f'Lagrange Interpolation n={n}', linestyle='--')
    plt.scatter(x_nodes, y_nodes, color='red')
    plt.title(f'Interpolation Comparison for n={n}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def chebyshev_nodes(n, xmin, xmax):
    return 0.5 * ((xmax - xmin) * np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * n + 2)) + (xmin + xmax))

def plot_chebyshev_interpolation(n):
    x_min, x_max = -5, 5
    x_nodes = chebyshev_nodes(n, x_min, x_max)
    y_nodes = f(x_nodes)
    
    x_dense = np.linspace(x_min, x_max, 400)
    y_dense = [lagrange_interpolation(x, x_nodes, y_nodes) for x in x_dense]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, f(x_dense), label='Original Function $f(x) = e^{-x^2}$')
    plt.plot(x_dense, y_dense, label=f'Chebyshev Lagrange Interpolation n={n}', linestyle='--')
    plt.scatter(x_nodes, y_nodes, color='red')
    plt.title(f'Chebyshev Interpolation Comparison for n={n}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example of plotting for n = 5, 10, 15, 20
for n in [5, 10, 15, 20]:
    plot_chebyshev_interpolation(n)

# for n in [5, 10, 15, 20]:
#     plot_interpolation(n)