import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")

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

    st.subheader("Classic Newton-Raphson")
    st.write("The plot shows the x_i and epsilon values for each iteration. As the epsilon values decrease, the algorithm converges to the root. The x_i value represents the root of the function after the iteration.")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(y_hist)
    with col2:
        st.line_chart(y_eps)

    print("classic: " + str(count))
    return x_next

def newton_method_double(x_0, resolution=4):
    count = 0

    x_hist = np.zeros([resolution])
    for i in range(resolution):
        x_hist[i] = i
    y_hist = np.zeros([resolution])
    y_eps = np.zeros([resolution])

    x_prev = x_0
    x_next = x_prev - (f(x_0) / derivative_f(x_0))

    while (abs(x_next - x_prev) > pow(10, -6) and count < resolution):
        y_hist[count] = x_next
        y_eps[count] = abs(x_next - x_prev)
        x_prev = x_next
        x_next = x_prev - 2 * (f(x_prev) / derivative_f(x_prev))
        count += 1

    st.subheader("Double Newton-Raphson")
    st.write("The plot shows the x_i and epsilon values for each iteration. As the epsilon values decrease, the algorithm converges to the root. The x_i value represents the root of the function after the iteration.")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(y_hist)
    with col2:
        st.line_chart(y_eps)

    print("double: " + str(count))
    return x_next

def u(x):
    return f(x) / derivative_f(x)

def derivative_u(x):
    return 1 - (((-1) * pow(x, -2) * pow(x - 2, 2) + (1 / x) * 2 * (x - 2) + (1 / x) * (2 * x - 4) + np.log(x) * 2) / derivative_f(x)) * u(x)

def newton_method_u(x_0, resolution=6):
    count = 0

    x_hist = np.zeros([resolution])
    for i in range(resolution):
        x_hist[i] = i
    y_hist = np.zeros([resolution])
    y_eps = np.zeros([resolution])

    x_prev = x_0
    x_next = x_prev - (f(x_0) / derivative_u(x_0))

    while (abs(x_next - x_prev) > pow(10, -6) and count < resolution):
        y_hist[count] = x_next
        y_eps[count] = abs(x_next - x_prev)
        x_prev = x_next
        x_next = x_prev - (u(x_prev) / derivative_u(x_prev))
        count += 1

    st.subheader("Newton-Raphson with u function")
    st.write("The plot shows the x_i and epsilon values for each iteration. As the epsilon values decrease, the algorithm converges to the root. The x_i value represents the root of the function after the iteration.")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(y_hist)
    with col2:
        st.line_chart(y_eps)

    print("with u func: " + str(count))
    return x_next

x_vals = np.linspace(0.5, 2.4, 100)
y_vals = f(x_vals)
plot_data = pd.DataFrame(y_vals, x_vals)
st.subheader("f(x)")
st.line_chart(plot_data)

slider_value = st.slider("Select a start point", 0.0, 2.4, 1.5)

print(newton_method(slider_value))
double_newton_res = st.slider("Select a resolution for the double Newton-Raphson", 1, 10, 4)
print(newton_method_double(slider_value, double_newton_res))
u_fun_newton_res = st.slider("Select a resolution for the u function Newton-Raphson", 1, 10, 6)
print(newton_method_u(slider_value, u_fun_newton_res))

