import numpy as np
import matplotlib.pyplot as plt

# Dane czasowe dla odpowiedzi na sygnał prostokątny filtru krytycznego
t_critical = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100])  # czas w mikrosekundach
Uout_critical_rect = np.array([4, 12, 24, 39, 47, 56, 64, 71, 78, 82, 86, 89, 94, 100])  # napięcie wyjściowe w mV

# Dane czasowe dla odpowiedzi na sygnał prostokątny filtru Butterwortha
t_butterworth = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 100])  # czas w mikrosekundach
Uout_butterworth_rect = np.array([4, 14, 29, 42, 56, 68, 77, 86, 91, 97, 100, 103, 104, 103, 102, 101, 101, 100])  # napięcie wyjściowe w mV

# Tworzenie wykresów odpowiedzi na sygnał prostokątny
plt.figure(figsize=(10, 6))

plt.plot(t_critical, Uout_critical_rect, label='Filtr krytyczny', marker='o', linestyle='-')
plt.plot(t_butterworth, Uout_butterworth_rect, label='Filtr Butterwortha', marker='s', linestyle='-')

plt.title('Odpowiedź filtrów na sygnał prostokątny')
plt.xlabel('Czas [µs]')
plt.ylabel('Napięcie wyjściowe [mV]')
plt.grid(True)
plt.legend()
plt.show()

