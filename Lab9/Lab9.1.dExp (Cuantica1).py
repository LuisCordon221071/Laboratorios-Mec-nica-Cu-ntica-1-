import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
energia1 = 1.0
energia2 = 2.0
acoplamiento = 0.5

# Configuración de la simulación
condiciones_iniciales = [
    {"a1": 1.0 + 0.0j, "a2": 0.0 + 0.0j, "etiqueta": "Caso 1: $a_1(0)=1$, $a_2(0)=0$"},
    {"a1": 0.0 + 0.0j, "a2": 1.0 + 0.0j, "etiqueta": "Caso 2: $a_1(0)=0$, $a_2(0)=1$"},
    {"a1": 1/np.sqrt(2) + 0.0j, "a2": 1/np.sqrt(2) + 0.0j, "etiqueta": "Caso 3: $a_1(0)=a_2(0)=1/\sqrt{2}$"}
]

tiempo_inicio = 0
tiempo_final = 10
paso_temporal = 0.01
pasos_totales = int((tiempo_final - tiempo_inicio) / paso_temporal)
tiempos = np.linspace(tiempo_inicio, tiempo_final, pasos_totales)

def paso_integracion(a1, a2, dt):
    def ecuaciones(a1, a2):
        derivada1 = -1j * (energia1 * a1 + acoplamiento * a2)
        derivada2 = -1j * (acoplamiento * a1 + energia2 * a2)
        return derivada1, derivada2
    
    k1_a1, k1_a2 = ecuaciones(a1, a2)
    k2_a1, k2_a2 = ecuaciones(a1 + 0.5*dt*k1_a1, a2 + 0.5*dt*k1_a2)
    k3_a1, k3_a2 = ecuaciones(a1 + 0.5*dt*k2_a1, a2 + 0.5*dt*k2_a2)
    k4_a1, k4_a2 = ecuaciones(a1 + dt*k3_a1, a2 + dt*k3_a2)
    
    nuevo_a1 = a1 + (dt/6) * (k1_a1 + 2*k2_a1 + 2*k3_a1 + k4_a1)
    nuevo_a2 = a2 + (dt/6) * (k1_a2 + 2*k2_a2 + 2*k3_a2 + k4_a2)
    return nuevo_a1, nuevo_a2

for caso in condiciones_iniciales:
    a1_actual, a2_actual = caso["a1"], caso["a2"]
    historial_a1 = np.zeros(pasos_totales, dtype=complex)
    historial_a2 = np.zeros(pasos_totales, dtype=complex)
    norma = np.zeros(pasos_totales)
    
    for paso in range(pasos_totales):
        historial_a1[paso] = a1_actual
        historial_a2[paso] = a2_actual
        norma[paso] = np.abs(a1_actual)**2 + np.abs(a2_actual)**2
        a1_actual, a2_actual = paso_integracion(a1_actual, a2_actual, paso_temporal)
    
    probabilidad1 = np.abs(historial_a1)**2
    probabilidad2 = np.abs(historial_a2)**2
    
    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, probabilidad1, label='$|a_1(t)|^2$', color='navy', linewidth=2)
    plt.plot(tiempos, probabilidad2, label='$|a_2(t)|^2$', color='crimson', linewidth=2)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    plt.title(caso["etiqueta"], fontsize=14, pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.text(0.02, 0.95, f'Conservación norma: {norma[-1]:.8f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{caso['etiqueta']}")
    print(f"Norma inicial: {norma[0]:.12f}")
    print(f"Norma final:   {norma[-1]:.12f}")