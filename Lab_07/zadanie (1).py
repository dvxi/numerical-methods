#!/usr/bin/env python3.11
import math
import numpy as np
import matplotlib.pyplot as plt
def main():
    n=65 #5,10,15,20
    #zmieniamy x-y z wielomianu Czebyszewa i znowu robimy interpolacje dla znowu 4 n roznych: 5,10,15,20
    x_min=-5
    x_max=5
    x_curr=x_min
    x1=[x_curr,]
    while x_curr<x_max:
        x_curr=x_curr+10/n
        x1.append(x_curr)
    """a=x_min
    b=x_max
    k = np.arange(0, n+1)
    print(k)
    x1 = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n+2))"""
        
    y1=[math.exp(-x**2) for x in x1]
    #mamy teraz n+1=11 wezlow i suzkamy wielomianu stopnia n=10
    x_axis=np.linspace(-5,5,200)
    print(Lagrange_function(x1, y1, n, 0) )
    y_axis=[Lagrange_function(x1, y1, n, x) for x in x_axis]
    fig,ax=plt.subplots()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    ax.plot(x_axis,np.exp(-x_axis**2),label=r'$f(x)=\;exp(-x^2)$')
    ax.plot(x_axis,y_axis, label='wielomian Lagrange\'a')
    plt.legend()
    plt.grid()
    plt.show()    


#mozna sprawdzic funkcje czy w wezle zwraca odpowiednia wartosc
def Lagrange_function(xs, ys, n,x):
    eps=1e-7
    licznik=1
    mianownik=1
    y=0
    for j in range(n+1):    #[0-n]
        for i in range(n+1):
            if i != j:
                licznik*=(x-xs[i])    
                mianownik*=(xs[j]-xs[i])    
        y+=ys[j]*licznik/mianownik
        licznik=1
        mianownik=1            
    return y

if __name__ == "__main__":
    main() 