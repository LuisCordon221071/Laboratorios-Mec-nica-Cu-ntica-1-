import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y  # Función actualizada
from matplotlib import cm

# Configuración de theta y phi (coordenadas esféricas)
theta = np.linspace(0, np.pi, 100)      # [0, π]
phi = np.linspace(0, 2 * np.pi, 100)    # [0, 2π]
Theta, Phi = np.meshgrid(theta, phi)    # Malla para evaluación

# Crear figura con subplots
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Armónicos Esféricos $Y_l^m(\\theta, \\phi)$ (Parte Real)', fontsize=16)

# Combinaciones (l, m) válidas para l=0,1,2
combinations = [
    (0, 0),
    (1, -1), (1, 0), (1, 1),
    (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
]

# Graficar cada Y_l^m
for idx, (l, m) in enumerate(combinations, start=1):
    ax = fig.add_subplot(3, 3, idx, projection='polar')
    
    # Calcular Y_l^m (sph_harm_y solo acepta m >= 0)
    Y_lm = sph_harm_y(abs(m), l, Phi, Theta)
    if m < 0:
        Y_lm = (-1)**abs(m) * np.conj(Y_lm)  # Ajuste para m negativo
    
    # Graficar la parte real
    c = ax.contourf(Phi, Theta, Y_lm.real, levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, shrink=0.8)
    
    ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', pad=15)
    ax.grid(True)

plt.tight_layout()
plt.show()