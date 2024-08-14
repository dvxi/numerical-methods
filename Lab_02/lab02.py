import numpy as np
np.set_printoptions(suppress=True)

def gaussian_elimination(mtrx, L, n):
    for i in range(n):
        for k in range(i+1, n):
            factor = mtrx[k, i] / mtrx[i, i]
            L[k,i] = factor
            for j in range(i, n):
                mtrx[k, j] -= factor * mtrx[i, j]
    return mtrx

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
    return y

n = 4

matrix = np.empty([n, n])

for i in range(n):
    for j in range(n):
        matrix[i,j] = (1.0 / (i + j + 2))

print("Macierz wejsciowa:")
print(matrix)

L = np.zeros([n,n])
for i in range(n):
    L[i,i] = 1

U = gaussian_elimination(matrix.copy(), L, n)
print("U:")
print(U)
print("L:")
print(L)
print("LU:")
print(np.matmul(L, U))

det_A = 1
for i in range(n):
    det_A *= U[i,i]
print("Wyznacznik macierzy A:")
print(det_A)

A_odwrotna = np.zeros([n, n])
for i in range(n):
    e_n = np.zeros(n)
    e_n[i] = 1
    col_n = gauss_jordan_elimination(matrix.copy(), e_n)
    for j in range(n):
        A_odwrotna[j, i] = col_n[j]

print("Macierz odwrotna do A")
print(A_odwrotna)

# Wskaznik uwarunkowania macierzy
print("Wskaznik uwarunkowania macierzy A:")
condA = 0
for i in range(n):
    for j in range(n):
        condA = max(condA, abs(matrix[i,j]))
print(condA)

print("Wskaznik uwarunkowania macierzy odwrotnej do A:")
condRevA = 0
for i in range(n):
    for j in range(n):
        condRevA = max(condRevA, abs(A_odwrotna[i,j]))
print(condRevA)