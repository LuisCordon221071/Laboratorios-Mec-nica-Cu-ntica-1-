import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ecuaciones_acopladas(t, a, E1, E2, V):
    a1, a2 = a
    derivada_a1 = -1j * (E1 * a1 + V * a2)
    derivada_a2 = -1j * (V * a1 + E2 * a2)
    return [derivada_a1, derivada_a2]

a1_inicial = 1.0 + 0.0j
a2_inicial = 0.0 + 0.0j

casos = [
    {"ΔE": 0.0, "V": 0.5, "etiqueta": "ΔE = 0 (Degenerado), V = 0.5"},
    {"ΔE": 1.0, "V": 0.1, "etiqueta": "ΔE = 1.0, V = 0.1 (Perturbación débil)"},
    {"ΔE": 1.0, "V": 0.5, "etiqueta": "ΔE = 1.0, V = 0.5"},
    {"ΔE": 0.5, "V": 1.0, "etiqueta": "ΔE = 0.5, V = 1.0 (Perturbación fuerte)"}
]

tiempo_inicio = 0
tiempo_fin = 10
tiempos = np.linspace(tiempo_inicio, tiempo_fin, 1000)

plt.figure(figsize=(12, 8))

for caso in casos:
    delta_E = caso["ΔE"]
    V = caso["V"]
    E1 = 1.0
    E2 = E1 + delta_E
    
    solucion = solve_ivp(
        ecuaciones_acopladas,
        [tiempo_inicio, tiempo_fin],
        [a1_inicial, a2_inicial],
        t_eval=tiempos,
        args=(E1, E2, V),
        method='RK45'
    )
    
    prob1 = np.abs(solucion.y[0])**2
    prob2 = np.abs(solucion.y[1])**2
    
    print(f"\nCaso: {caso['etiqueta']}")
    print("Norma inicial:", np.abs(a1_inicial)**2 + np.abs(a2_inicial)**2)
    print("Norma final:", prob1[-1] + prob2[-1])
    
    plt.plot(solucion.t, prob2, label=f"{caso['etiqueta']} | Máx $P_2$ = {np.max(prob2):.2f}")

plt.xlabel('Tiempo ($t$)')
plt.ylabel('Probabilidad $|a_2(t)|^2$')
plt.title('Evolución de $|a_2(t)|^2$ para distintos ΔE y V')
plt.legend()
plt.grid(True)
plt.show()