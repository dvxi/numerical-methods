import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
X_MIN = st.slider("Minimum value of x", -10, 0, -5, key='X_MIN')
X_MAX = st.slider("Maximum value of x", 0, 10, 5, key='X_MAX')

def f(x):
    return np.exp(-x**2)

def lagrange_basis(x, i, nodes):
    L = 1
    for j in range(len(nodes)):
        if j != i:
            L *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return L

def lagrange_interpolation(x, nodes, values):
    P = 0
    for i in range(len(values)):
        P += values[i] * lagrange_basis(x, i, nodes)
    return P

def plot_interpolation():
    st.header(f"Lagrange Interpolation")
    n = st.slider("Number of nodes", 1, 20, 5, key='interpolation')

    x_nodes = np.linspace(X_MIN, X_MAX, n+1)
    y_nodes = f(x_nodes)
    
    x_dense = np.linspace(X_MIN, X_MAX, 400)
    y_dense = [lagrange_interpolation(x, x_nodes, y_nodes) for x in x_dense]
    
    interpolation_dataframe = pd.DataFrame({
        'x': x_dense,
        'Original Function': f(x_dense),
        f'Lagrange Interpolation n={n}': y_dense
    })
    node_dataframe = pd.DataFrame({
        'x': x_nodes,
        'y': y_nodes
    })
    fig1 = px.line(interpolation_dataframe, x='x', y=['Original Function', f'Lagrange Interpolation n={n}'])
    fig2 = px.scatter(node_dataframe, x='x', y='y')

    fig1.add_trace(fig2.data[0])

    st.plotly_chart(fig1, use_container_width=True)

def chebyshev_nodes(n, xmin, xmax):
    return 0.5 * ((xmax - xmin) * np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * n + 2)) + (xmin + xmax))

def plot_chebyshev_interpolation():

    st.header(f"Chebyshev Interpolation")
    n = st.slider("Number of nodes", 1, 20, 5, key='chebyshev')

    x_nodes = chebyshev_nodes(n, X_MIN, X_MAX)
    y_nodes = f(x_nodes)
    
    x_dense = np.linspace(X_MIN, X_MAX, 400)
    y_dense = [lagrange_interpolation(x, x_nodes, y_nodes) for x in x_dense]
    
    # Assuming x_dense, f, y_dense, x_nodes, y_nodes, and n are defined

    interpolation_dataframe = pd.DataFrame({
        'x': x_dense,
        'Original Function': f(x_dense),
        f'Lagrange Interpolation n={n}': y_dense
    })
    node_dataframe = pd.DataFrame({
        'x': x_nodes,
        'y': y_nodes
    })
    fig1 = px.line(interpolation_dataframe, x='x', y=['Original Function', f'Lagrange Interpolation n={n}'])
    fig2 = px.scatter(node_dataframe, x='x', y='y')

    fig1.add_trace(fig2.data[0])

    st.plotly_chart(fig1, use_container_width=True)

st.title("Lagrange Interpolation")

col1, col2 = st.columns(2)
with col1:
    plot_interpolation()
with col2:
    plot_chebyshev_interpolation()