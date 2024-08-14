import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

k_ch = 592 #5.92 * 10^2 kg/(s^2)
k_cc = 1580 #1.58 * 10^2 kg/(s^2)

amu = 1.6605 * pow(10,-27)

m_h = 1
m_c = 12

N = 4

def create_matrix():
    matrix = np.zeros([4, 4])
    matrix[0][0] = k_ch / m_h
    matrix[1][0] = -k_ch / m_c
    matrix[0][1] = -k_ch / m_h
    matrix[1][1] = (k_ch + k_cc) / m_c
    matrix[2][2] = matrix[1][1]
    matrix[2][1] = -k_cc / m_c
    matrix[1][2] = matrix[2][1]
    matrix[2][3] = matrix[1][0]
    matrix[3][2] = matrix[0][1]
    matrix[3][3] = matrix[0][0]
    return matrix

def create_matrix_one():
    matrix = np.zeros([4, 4])
    for i in range(4):
        matrix[i][i] = 1
    return matrix

A = create_matrix()

# Rozklad QR metoda Hauseholdera

Q = create_matrix_one()
R = A

for i in range(N-1):
    x = np.zeros([N, 1])
    for j in range(i, N):
        x[j][0] = R[j][i]

    x_norm = np.linalg.norm(x)
    e = np.zeros([N, 1])
    e[i][0] = 1

    u = x - (x_norm * e)
    v = u / np.linalg.norm(u)

    I = create_matrix_one()

    Q_t = I - 2 * np.matmul(v, v.transpose())
    Q = np.matmul(Q_t, Q)
    R = np.matmul(Q_t, R)

Q = Q.transpose()

print(R)
print(np.matmul(Q, R))
print(A)