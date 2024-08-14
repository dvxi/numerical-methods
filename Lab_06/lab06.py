import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def f(x):
    return np.log(x) * pow((x - 2), 2)

def derivative_f(x):
    return ((1 / x) * pow(x - 2, 2)) + (np.log(x) * (2*x - 4))

def newton_method(x_0):
    count = 0

    x_hist = np.zeros([17])
    for i in range(17):
        x_hist[i] = i
    y_hist = np.zeros([17])
    y_eps = np.zeros([17])

    x_prev = x_0
    x_next = x_prev - (f(x_0) / derivative_f(x_0))

    while (abs(x_next - x_prev) > pow(10, -6)):
        y_hist[count] = x_next
        y_eps[count] = abs(x_next - x_prev)
        x_prev = x_next
        x_next = x_prev - (f(x_prev) / derivative_f(x_prev))
        count += 1

    plt.title("Classic - x_i & epsilon")
    plt.plot(x_hist, y_hist)
    plt.plot(x_hist, y_eps)
    plt.show()

    print("classic: " + str(count))
    return x_next

def newton_method_double(x_0):
    count = 0

    x_hist = np.zeros([4])
    for i in range(4):
        x_hist[i] = i
    y_hist = np.zeros([4])
    y_eps = np.zeros([4])

    x_prev = x_0
    x_next = x_prev - (f(x_0) / derivative_f(x_0))

    while (abs(x_next - x_prev) > pow(10, -6)):
        y_hist[count] = x_next
        y_eps[count] = abs(x_next - x_prev)
        x_prev = x_next
        x_next = x_prev - 2 * (f(x_prev) / derivative_f(x_prev))
        count += 1

    plt.title("Double - x_i & epsilon")
    plt.plot(x_hist, y_hist)
    plt.plot(x_hist, y_eps)
    plt.show()

    print("double: " + str(count))
    return x_next

def u(x):
    return f(x) / derivative_f(x)

def derivative_u(x):
    return 1 - (((-1) * pow(x, -2) * pow(x - 2, 2) + (1 / x) * 2 * (x - 2) + (1 / x) * (2 * x - 4) + np.log(x) * 2) / derivative_f(x)) * u(x)

def newton_method_u(x_0):
    count = 0

    x_hist = np.zeros([6])
    for i in range(6):
        x_hist[i] = i
    y_hist = np.zeros([6])
    y_eps = np.zeros([6])

    x_prev = x_0
    x_next = x_prev - (f(x_0) / derivative_u(x_0))

    while (abs(x_next - x_prev) > pow(10, -6)):
        y_hist[count] = x_next
        y_eps[count] = abs(x_next - x_prev)
        x_prev = x_next
        x_next = x_prev - (u(x_prev) / derivative_u(x_prev))
        count += 1

    plt.title("Method u(x) - x_i & epsilon")
    plt.plot(x_hist, y_hist)
    plt.plot(x_hist, y_eps)
    plt.show()

    print("with u func: " + str(count))
    return x_next

x_vals = np.linspace(0.5, 2.4, 100)
y_vals = f(x_vals)

print(newton_method(1.5))
print(newton_method_double(1.5))
print(newton_method_u(1.5))

plt.title("f(x)")
plt.plot(x_vals, y_vals)
plt.show()