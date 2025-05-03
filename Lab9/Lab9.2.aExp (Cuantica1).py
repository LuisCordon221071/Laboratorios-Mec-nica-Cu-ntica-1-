import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Configuración del sistema
energia_base = 1.0
energia_excitado = 2.0
frecuencia = 1.0
tiempo_simulacion = [0, 20]
amplitudes = [0.5, 1.0, 3.0]

def calcular_perturbacion(t, amplitud, omega):
    return amplitud * np.cos(omega * t)

def ecuaciones_sistema(t, estado, E1, E2, amplitud, omega):
    a1, a2 = estado
    perturbacion = calcular_perturbacion(t, amplitud, omega)
    derivada_a1 = -1j * (E1 * a1 + perturbacion * a2)
    derivada_a2 = -1j * (perturbacion * a1 + E2 * a2)
    return [derivada_a1, derivada_a2]

for amplitud in amplitudes:
    plt.figure(figsize=(10, 5))
    
    solucion = solve_ivp(
        ecuaciones_sistema,
        tiempo_simulacion,
        [1.0 + 0.0j, 0.0 + 0.0j],
        args=(energia_base, energia_excitado, amplitud, frecuencia),
        t_eval=np.linspace(*tiempo_simulacion, 1000),
        method='RK45'
    )
    
    prob_estado1 = np.abs(solucion.y[0])**2
    prob_estado2 = np.abs(solucion.y[1])**2
    norma_total = prob_estado1 + prob_estado2

    plt.plot(solucion.t, prob_estado1, label='Probabilidad estado base', color='blue')
    plt.plot(solucion.t, prob_estado2, label='Probabilidad estado excitado', color='red')
    plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.text(0.02, 0.95, f'Conservación norma: {norma_total[-1]:.6f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Dinámica del sistema con $V_0={amplitud}$')
    plt.xlabel('Tiempo (unidades arbitrarias)')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.show()

    print(f"\nResultados para V0 = {amplitud}:")
    print(f"Máxima transferencia al estado excitado: {np.max(prob_estado2):.2%}")
    print(f"Norma conservada (promedio): {np.mean(norma_total):.8f}")