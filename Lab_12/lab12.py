import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
np.set_printoptions(suppress=True)

def f(x, m, k):
    return np.power(x, m) * np.sin(k * x)

def simp_integral(f, x_array, h, m, k, n):
    sum = 0
    for i in range(1, int(n / 2) + 1):
        sum += f(x_array[2 * i - 2], m, k) + 4 * f(x_array[2 * i - 1], m, k) + f(x_array[2 * i], m, k)
    sum *= (h / 3)
    return sum

def series_integral(k, m, b, a, table = False):
    sum = 0
    for i in range(30):
        sum += pow(-1, i) * (pow(k * b, 2 * i + m + 2)) / (pow(k, m + 1) * math.factorial(2 * i + 1) * (2 * i + m + 2))
        sum -= pow(-1, i) * (pow(k * a, 2 * i + m + 2)) / (pow(k, m + 1) * math.factorial(2 * i + 1) * (2 * i + m + 2))
        if(table != False):
            table.append(sum)
    return sum

N = 101
B = np.pi
A = 0
H = (B - A) / N

M = 5
K = 5

x_array = np.linspace(0, np.pi, N)
y_array = f(x_array, M, K)

plt.plot(x_array, y_array)
plt.title("f(x)")
plt.show()

# 1

## a

M = 0
K = 1

sumTable = []
series_int = series_integral(K, M, B, A, sumTable)

print(series_int)
plt.plot(sumTable)
plt.title("Series integral m=0 k=1")
plt.show()

## b

M = 1
K = 1

sumTable = []
series_int = series_integral(K, M, B, A, sumTable)

print(series_int)
plt.plot(sumTable)
plt.title("Series integral m=1 k=1")
plt.show()

## c

M = 5
K = 5

sumTable = []
series_int = series_integral(K, M, B, A, sumTable)

print(series_int)
plt.plot(sumTable)
plt.title("Series integral m=5 k=5")
plt.show()

# 2

nTable = [11, 21, 51, 101, 201]

CITable_1 = []
CITable_2 = []
CITable_3 = []

for N in nTable:
    x_array = np.linspace(0, np.pi, N)

    H = (B - A) / N

    M = 0
    K = 1

    simpson_int = simp_integral(f, x_array, H, M, K, N - 1)
    series_int = series_integral(K, M, B, A)
    CITable_1.append(abs(simpson_int - series_int))
    print("m=0, k=1, Simpson: " + str(simpson_int) + ", N=" + str(N))

    M = 1
    K = 1

    simpson_int = simp_integral(f, x_array, H, M, K, N - 1)
    series_int = series_integral(K, M, B, A)
    CITable_2.append(abs(simpson_int - series_int))
    print("m=1, k=1, Simpson: " + str(simpson_int) + ", N=" + str(N))

    M = 5
    K = 5

    simpson_int = simp_integral(f, x_array, H, M, K, N - 1)
    series_int = series_integral(K, M, B, A)
    CITable_3.append(abs(simpson_int - series_int))
    print("m=5, k=5, Simpson: " + str(simpson_int) + ", N=" + str(N))

plt.plot(nTable, CITable_1)
plt.title("|C-I| m=0 k=1")
plt.show()

plt.plot(nTable, CITable_2)
plt.title("|C-I| m=1 k=1")
plt.show()

plt.plot(nTable, CITable_3)
plt.title("|C-I| m=5 k=5")
plt.show()

# series_int = series_integral(K, M, B, A)
# print(series_int)