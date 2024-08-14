import numpy as np
import matplotlib.pyplot as plt

def generate_solvable_matrix(n):
    while True:
        mtrx = np.random.rand(n, n)
        if np.linalg.det(mtrx) != 0:  # Ensure the matrix is non-singular
            return mtrx

n = 6
mtrx = generate_solvable_matrix(n)
y = np.random.rand(n)

print("Solvable matrix:")
print(mtrx)
print("y vector:")
print(y)

# Perform the Gaussian elimination
def gaussian_elimination(mtrx, y):
    n = len(y)
    
    # Forward elimination to form upper triangular matrix
    for i in range(n):
        # Pivot for mtrx[i][i]
        for k in range(i+1, n):
            factor = mtrx[k, i] / mtrx[i, i]
            for j in range(i, n):
                mtrx[k, j] -= factor * mtrx[i, j]
            y[k] -= factor * y[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= mtrx[i, j] * x[j]
        x[i] = x[i] / mtrx[i, i]

    print("Matrix after Gaussian elimination:")
    print(mtrx)
    return x

# Perform the Gauss-Jordan elimination
def gauss_jordan_elimination(mtrx, y):
    n = len(y)
    for i in range(n):
        factor = mtrx[i, i]
        mtrx[i] /= factor
        y[i] /= factor
        
        for j in range(n):
            if i != j:
                factor = mtrx[j, i]
                mtrx[j] -= factor * mtrx[i]
                y[j] -= factor * y[i]
    print("Matrix after Gauss-Jordan elimination:")
    print(mtrx)
    return y

print("Solution using Gaussian elimination:")
result_gauss = gaussian_elimination(mtrx.copy(), y.copy())
print(result_gauss)
print("Solution using Gauss-Jordan elimination:")
reuslt_gauss_jordan = gauss_jordan_elimination(mtrx.copy(), y.copy())
print(reuslt_gauss_jordan)

# Displaying the results

x_range = np.linspace(result_gauss[0], result_gauss[-1], 100)
c = result_gauss

def horner(c, x):
    w = c[-1]
    for i in range(len(c) - 2, -1, -1):
        w = w * x + c[i]
    return w

y_range = [horner(c, x) for x in x_range]

plt.plot(x_range, y_range, label='Wartości wielomianu')
plt.show()