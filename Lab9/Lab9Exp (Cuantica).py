import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
E1 = 1.0
E2 = 2.0
V = 0.5

# Condiciones iniciales
a1_inicial = 1.0 + 0.0j
a2_inicial = 0.0 + 0.0j

# Configuración temporal
tiempo_inicio = 0
tiempo_fin = 10
paso = 0.01
num_pasos = int((tiempo_fin - tiempo_inicio) / paso)

def ecuaciones(a1, a2):
    derivada_a1 = -1j * (E1 * a1 + V * a2)
    derivada_a2 = -1j * (V * a1 + E2 * a2)
    return derivada_a1, derivada_a2

def paso_rk4(a1, a2, paso):
    k1_a1, k1_a2 = ecuaciones(a1, a2)
    k2_a1, k2_a2 = ecuaciones(a1 + 0.5 * paso * k1_a1, a2 + 0.5 * paso * k1_a2)
    k3_a1, k3_a2 = ecuaciones(a1 + 0.5 * paso * k2_a1, a2 + 0.5 * paso * k2_a2)
    k4_a1, k4_a2 = ecuaciones(a1 + paso * k3_a1, a2 + paso * k3_a2)
    
    nuevo_a1 = a1 + (paso / 6) * (k1_a1 + 2*k2_a1 + 2*k3_a1 + k4_a1)
    nuevo_a2 = a2 + (paso / 6) * (k1_a2 + 2*k2_a2 + 2*k3_a2 + k4_a2)
    return nuevo_a1, nuevo_a2

# Simulación
tiempos = np.linspace(tiempo_inicio, tiempo_fin, num_pasos)
valores_a1 = np.zeros(num_pasos, dtype=complex)
valores_a2 = np.zeros(num_pasos, dtype=complex)
normas = np.zeros(num_pasos)

a1, a2 = a1_inicial, a2_inicial
for i in range(num_pasos):
    valores_a1[i] = a1
    valores_a2[i] = a2
    normas[i] = np.abs(a1)**2 + np.abs(a2)**2
    a1, a2 = paso_rk4(a1, a2, paso)

# Resultados
probabilidad1 = np.abs(valores_a1)**2
probabilidad2 = np.abs(valores_a2)**2

print("Norma inicial:", normas[0])
print("Norma final:", normas[-1])

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(tiempos, probabilidad1, label='$|a_1(t)|^2$')
plt.plot(tiempos, probabilidad2, label='$|a_2(t)|^2$')
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.title('Evolución del Sistema Cuántico')
plt.legend()
plt.grid(True)
plt.show()