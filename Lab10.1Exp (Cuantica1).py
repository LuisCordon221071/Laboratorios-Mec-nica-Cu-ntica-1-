import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv  # Polinomios asociados de Legendre

def psi_n(x, a, V0, n, M=1.0, hbar=1.0):

    lambda_sq = (2 * M * V0 * a**2) / (hbar**2)
    s = np.sqrt(lambda_sq + 0.25) - 0.5
    m = s - n
    
    if m < 0:
        raise ValueError("n no puede ser mayor que s (n < s).")
    
    xi = np.tanh(x / a)
    P = lpmv(m, s, xi)  # Polinomio asociado de Legendre
    psi = (1 - xi**2)**(m / 2) * P
    
    # Normalización (aproximada)
    N_n = 1 / np.sqrt(a * np.trapz(np.abs(psi)**2, x))
    return N_n * psi

# Parámetros del pozo
a = 1.0
V0 = 10.0
x = np.linspace(-5, 5, 1000)

# Graficar los primeros 3 estados
plt.figure(figsize=(10, 6))
for n in range(3):
    psi = psi_n(x, a, V0, n)
    plt.plot(x, psi, label=f"n = {n}")

plt.title("Funciones de onda estacionarias $\psi_n(x)$")
plt.xlabel("x")
plt.ylabel("$\psi_n(x)$")
plt.legend()
plt.grid()
plt.show()