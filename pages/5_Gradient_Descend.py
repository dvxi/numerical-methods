import numpy as np
from numpy.linalg import inv
import plotly.express as px
import streamlit as st

DELTA = pow(10, -4)

st.set_page_config(layout="wide")
st.title("Gradien Descend for searching minimum of a 2D function")

col1, col2 = st.columns(2)

with col1:
    st.latex(r"f(x, y) = \frac{5}{2} \cdot (x^2 - y)^2 + (1 - x)^2")
    H = st.slider("Step size (H)", min_value=0.001, max_value=0.1, value=0.1)
    X_MIN = st.slider("Minimum X value", min_value=-10.0, max_value=10.0, value=-2.0)
    X_MAX = st.slider("Maximum X value", min_value=-10.0, max_value=10.0, value=2.0)
    Y_MIN = st.slider("Minimum Y value", min_value=-10.0, max_value=10.0, value=-2.0)
    Y_MAX = st.slider("Maximum Y value", min_value=-10.0, max_value=10.0, value=2.0)
    CONTOUR_STEP = st.slider("Contour Step", min_value=0.1, max_value=1.0, value=0.5)
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        START_X = st.number_input(f"Starting X value between {X_MIN} and {X_MAX}", min_value=X_MIN, max_value=X_MAX, value=0.0)
    with in_col2:
        START_Y = st.number_input(f"Starting Y value between {Y_MIN} and {Y_MAX}", min_value=Y_MIN, max_value=Y_MAX, value=0.0)

def f_xy(x, y):
    return (5/2) * pow((pow(x, 2) - y), 2) + pow(1 - x, 2)

def grad_f(f, x, y):
    f_x = (f(x + DELTA, y) - f(x - DELTA, y)) / (2 * DELTA)
    f_y = (f(x, y + DELTA) - f(x, y - DELTA)) / (2 * DELTA)

    return np.array([f_x, f_y])

def vec_dist(r_0, r_1):
    return np.sqrt(pow(r_0[0] - r_1[0], 2) + pow(r_0[1] - r_1[1], 2))

def find_min(f, x_0, y_0, eps):

    if(eps == 0.01):
        file = open("eps1.dat", 'w')
        file.write("--- Epsilon = 10^-2 ---\n")
        eps01_path.fill(0)
    else:
        file = open("eps2.dat", "w")
        file.write("--- Epsilon = 10^-3 ---\n")
        eps01_path.fill(0)

    r_i = np.array([x_0, y_0])
    for i in range(1000):
        if(eps == 0.01):
            eps01_path[i] = r_i
        else:
            eps001_path[i] = r_i

        r_i_1 = r_i - H * grad_f(f, r_i[0], r_i[1])
        print("x: " + str(r_i[0]) + ", y: " + str(r_i[1]))

        file.write("x: " + str(r_i[0]) + ", y: " + str(r_i[1]) + "\n")

        if(vec_dist(r_i, r_i_1) < eps):
            print("Spelnione po " + str(i + 1) + " iteracjach.")
        r_i = r_i_1
    
    file.close()

print("--- Epsilon = 10^-2 ---")
eps01_path = np.zeros((1000, 2))
find_min(f_xy, -0.75, 1.75, 0.01)

# print("Path of 0.01 path:")
# for row in eps01_path:
#     print(row)

print("--- Epsilon = 10^-3 ---")
eps001_path = np.zeros((1000, 2))
find_min(f_xy, START_X, START_Y, 0.001)

x = np.linspace(X_MIN, X_MAX, 100)
y = np.linspace(Y_MIN, Y_MAX, 100)
X, Y = np.meshgrid(x, y)
Z = f_xy(X, Y)

Z_max = np.max(Z)
Z_min = np.min(Z)

z_scale = np.linspace(Z_min, Z_max, 10)

with col2:
    fig = px.imshow(Z, x=x, y=y, color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title='X',
        yaxis_title='Y')

    fig.add_scatter(x=eps001_path[:, 0], y=eps001_path[:, 1], mode='lines', line=dict(color='red'), name='eps: 0.001 path')

    fig.add_contour(x=x, y=y, z=Z, contours=dict(start=min(z_scale), end=max(z_scale), size=1), colorscale='Viridis')
    fig.update_layout(height=800, width=800)  # Adjust the height and width as desired

    fig.update_coloraxes(showscale=False)

    st.plotly_chart(fig)