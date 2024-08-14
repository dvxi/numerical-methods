import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

T = 1.0
T_MAX = 3.0 * T
SIGMA = T / 20.0
OMEGA = (2.0 * np.pi) / T

def delta():
    return np.random.rand() - 0.5

def N_k(k):
    return pow(2, k)

def f(t):
    return np.sin(1.0 * OMEGA * t) + np.sin(2.0 * OMEGA * t) + np.sin(3.0 * OMEGA * t)

def g(t):
    return (1.0 / (SIGMA * np.sqrt(2.0 * np.pi))) * np.exp(-pow(t, 2) / (2.0 * pow(SIGMA, 2)))

# Program

Ns = [N_k(8), N_k(10), N_k(12)]

for N in Ns:

    # Define time axis
    x_axis = np.linspace(0, T_MAX, N)  # Start from 0 for proper Gaussian centering
    y_axis = f(x_axis)
    g_filter = g(x_axis)

    # Plot original signal
    plt.plot(x_axis, y_axis)
    plt.title("Original Signal f(t)")
    plt.show()

    # Add noise to the signal
    y_axis_noise = y_axis + np.random.uniform(-0.5, 0.5, size=N)

    # Plot noisy signal
    plt.plot(x_axis, y_axis_noise)
    plt.title("Noisy Signal f(t) + noise")
    plt.show()

    # Perform FFT on the noisy signal and the Gaussian filter
    f_fft = np.fft.fft(y_axis_noise)
    g_fft = np.fft.fft(g_filter)
    g_2_fft = np.fft.ifft(g_filter)

    # Apply the Gaussian filter in the frequency domain
    f_filtered_fft = f_fft * (g_fft + g_2_fft)

    # Inverse FFT to get the filtered time-domain signal
    f_filtered = np.fft.ifft(f_filtered_fft)

    # Normalize the function

    f_max = f_filtered[0]
    for i in range(N):
        if(abs(f_filtered[i]) > abs(f_max)):
            f_max = f_filtered[i]

    f_filtered *= 2.5 / abs(f_max)

    # Plot the denoised signal
    plt.plot(x_axis, f_filtered.real)  # Use the real part of the result
    plt.title("Denoised Signal")
    plt.show()
