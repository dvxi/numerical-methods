import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def create_matrix(n, m):
    matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(np.clip(i-m, 0, n), np.clip(i+m+1, 0, n)):
            matrix[i][j] = 1 / (1 + np.abs(i - j))
    return matrix

def fastest_decline(A, b, x):
    r = np.subtract(b, np.matmul(A, x))
    data = open("data_log.txt", "w")
    # data.write("r;x;\n")
    data.write(str(np.sqrt(np.matmul(r, np.transpose(r)))))
    data.write(";")
    data.write(str(np.sqrt(np.matmul(x, np.transpose(x)))))
    data.write("\n")
    while(np.sqrt(np.matmul(r, np.transpose(r))) > 10**(-6) ):
        print(np.sqrt(np.matmul(r, np.transpose(r))))
        a = np.matmul(r, np.transpose(r)) / np.matmul(np.transpose(r), np.matmul(r, A))
        x = np.add(x, np.multiply(a, r))
        r = np.subtract(b, np.matmul(A, x))
        data.write(str(np.sqrt(np.matmul(r, np.transpose(r)))))
        data.write(";")
        data.write(str(np.sqrt(np.matmul(x, np.transpose(x)))))
        data.write("\n")
    data.close()
    return r, x

n = 1000
m = 5

A = create_matrix(n, m)
# print(A)

b = np.zeros([n])
for i in range(n):
    b[i] = i

# x = 0
x = np.zeros([n])
for i in range(n):
    x[i] = 0

r, x = fastest_decline(A, b, x)
print("Wektor reszt:")
print(r)
print("Wektor rozwiazan")
print(x)

rowsCount = len(open("data_log.txt", 'r').readlines())
print("Ilosc wierszy:\n")
print(rowsCount)
xAxis = np.zeros(rowsCount)
for i in range(rowsCount):
    xAxis[i] = i
yAxis1 = np.zeros(rowsCount)
yAxis2 = np.zeros(rowsCount)

with open("data_log.txt", 'r') as data:
    i = 0
    for line in data:
        print(line)
        xElem, yElem = line.split(";")
        yAxis1[i] = xElem
        yAxis2[i] = yElem
        i+=1
        print(xAxis)

# axis = plt.subplot(2)

# axis[0,0].plot(xAxis, yAxis1)
# axis[0,0].set_title("||r||")

# axis[1,0].plot(xAxis, yAxis2)
# axis[1,0].set_title("||x||")

plt.plot(xAxis, yAxis1)
plt.plot(xAxis, yAxis2)
plt.show()