import numpy as np
from numpy.linalg import inv
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Spline Interpolation")

N = 5
X_0 = -5
X_1 = 5
DX = 0.01

col1, col2, col3 = st.columns(3)

with col1:
    st.latex(r"f(x) = \frac{1}{1 + x^2}")
    N = st.slider("Number of points (N)", min_value=2, max_value=100, value=5)
    X_0 = st.slider("Start value (X_0)", min_value=-10, max_value=0, value=-5)
    X_1 = st.slider("End value (X_1)", min_value=0, max_value=10, value=5)
    DX = st.slider("Step size (DX)", min_value=0.01, max_value=1.0, value=0.01)

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

xw = np.linspace(X_0, X_1, N)
yw = f(xw)

print("xw:")
print(xw)

m = wyzM(xw, yw, N, 0, 0)
sx = wyzSx(xw, yw, m, N, -1.0)

print("m:")
print(m)

print("sx:")
print(sx)

x_new = np.linspace(X_0, X_1, N)
x_smooth = np.linspace(X_0, X_1, 200)
y_new = f(x_new)

i = 0
for iks in x_new:
    y_new[i] = wyzSx(xw, yw, m, 100, iks)

with col2:
    st.header("Spline interpolation compared to original function")
    st.write("The plot shows the original f(x) function and the interpolated function using spline interpolation.")

    # Create a DataFrame with the data
    df = pd.DataFrame({
        'x': x_smooth.tolist() + x_new.tolist(),
        'y': f(x_smooth).tolist() + y_new.tolist(),
        'type': ['Original Function'] * len(x_smooth) + ['Interpolated Function'] * len(x_new)
    })

    # Create the figure
    fig = px.line(df, x='x', y='y', color='type')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Drugie pochodne
    delta_x = 0.01
    approx_second_derivatives = [wyzApproxSecondDerivative(f, x, delta_x) for x in xw]
    exact_second_derivatives = [f2(x) for x in xw]

    derivatives_dataframe = pd.DataFrame({
        'x': xw,
        'Approximated Second Derivatives': approx_second_derivatives,
        'Exact Second Derivatives': exact_second_derivatives
    })

    st.write("Comparison of approximated and exact second derivatives. Analitycally calculated second derivative is equal to:")
    st.latex(r"f_2(x) = \cos(2x)")
    fig = px.line(derivatives_dataframe, x='x', y=['Approximated Second Derivatives', 'Exact Second Derivatives'])
    st.plotly_chart(fig)

with col3:
    # Interpolacja dla f1(x) oraz f2(x) w przedziale x ∈ [−5,5], dla liczby węzłów: n = 5, 8, 21
    n = st.slider("Number of nodes", min_value=2, max_value=100, value=5)
    xw = np.linspace(X_0, X_1, n)
    yw = f(xw)
    m = wyzM(xw, yw, n, 0, 0)
    x_new = np.linspace(X_0, X_1, 100)
    y_new_interpolated = [wyzSx(xw, yw, m, n, x) for x in x_new]

    interpol_dataframe = pd.DataFrame({
        'x': x_new,
        'Original Function': f(x_new),
        f'Interpolated Function n={n}': y_new_interpolated
    })

    fig = px.line(interpol_dataframe, x='x', y=['Original Function', f'Interpolated Function n={n}'])
    st.plotly_chart(fig)

    xw = np.linspace(-5, 5, n)
    yw = f2(xw)
    m = wyzM(xw, yw, n, 0, 0)
    x_new = np.linspace(-5, 5, 100)
    y_new_interpolated = [wyzSx(xw, yw, m, n, x) for x in x_new]

    interpol_f2_dataframe = pd.DataFrame({
        'x': x_new,
        'Original Function': f(x_new),
        f'Interpolated Function n={n}': y_new_interpolated
    })

    fig = px.line(interpol_f2_dataframe, x='x', y=['Original Function', f'Interpolated Function n={n}'])
    st.plotly_chart(fig)
