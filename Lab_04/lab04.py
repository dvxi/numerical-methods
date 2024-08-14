import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# IT_MAX = 50
# N = 50
# L = 5
# dx = 2L/N = 10/50 = 1/5
# xi = -L + i*dx = -5 + 1/5 * i

def create_matrix(n, l):
    matrix = np.zeros([n, n])
    dx = 2*l / n
    for i in range(n):
        xi = -l + i * dx
        for j in range(np.clip(i-1, 0, n), np.clip(i+1+1, 0, n)):
            if(i == j):
                matrix[i][i] = pow(dx, -2) + (pow(xi, 2) / 2)
            else:
                matrix[i][j] = -1 / (2 * pow(dx, 2))
    return matrix

def max_row_sum(matrix, N):
    S = 0
    for i in range(N):
        S += abs(matrix[0][i])

    for i in range(N):
        currS = 0
        for j in range(N):
            currS += abs(matrix[i][j])
        if(currS > S):
            S = currS
    return S

def find_lbd(matrix, IT_MAX, N, i_val):
    S = max_row_sum(matrix, N)
    a = -S #najw suma modulow w wierszu
    b = S

    for i in range(IT_MAX):
        lbd = (a + b) / 2
        # print(lbd)
        p = np.zeros([N])
        p[0] = 1
        p[1] = matrix[0][0] - lbd

        sgn_count = 0

        for j in range(2, N):
            p[j] = (matrix[j][j] - lbd) * p[j-1] + pow(matrix[j, j-1], 2) * p[j - 2]
            if(p[j] * p[j-1] < 0):
                sgn_count += 1
        
        print(str(sgn_count) + " | " + str(i_val) + " a:" + str(a) + " b: " + str(b) + " lbd: " + str(lbd))
        if(sgn_count <= i_val):
            a = lbd
        else:
            b = lbd
    return lbd

def find_eigenvector(lbd, matrix):
    N = matrix.shape[0]
    x = np.zeros(N)
    x[0] = 1  # x_j^1 = 1
    # Adjusted formula for zero-based index
    x[1] = (lbd - matrix[0, 0]) / matrix[1, 0]
    for n in range(2, N):
        x[n] = ((lbd - matrix[n, n]) * x[n - 1] - matrix[n-1, n-2] * x[n - 2]) / matrix[n, n-1]
    return x

matrix = create_matrix(50, 5)
# mtx = create_matrix(5,5)
print(matrix)

lambdas = np.zeros(5)

lambdas[0] = find_lbd(matrix, 50, 50, 1)
lambdas[1] = find_lbd(matrix, 50, 50, 2)
lambdas[2] = find_lbd(matrix, 50, 50, 3)
lambdas[3] = find_lbd(matrix, 50, 50, 4)
lambdas[4] = find_lbd(matrix, 50, 50, 5)

plt.figure(figsize=(10, 8))
for i, lbd in enumerate(lambdas, start=1):
    print("lambda " + str(i) + ": " + str(lbd))
    eigenvector = find_eigenvector(lbd, matrix)
    plt.plot(eigenvector, label=f'Lambda {i}')

plt.title('Eigenvectors for the First Five Eigenvalues')
plt.xlabel('Index')
plt.ylabel('Eigenvector Component Value')
plt.legend()
plt.show()

# print("1: " + str(lbd1) + ", 2: " + str(lbd2) + ", 3: " + str(lbd3) + ", 4: " + str(lbd4) + ", 5: " + str(lbd5))