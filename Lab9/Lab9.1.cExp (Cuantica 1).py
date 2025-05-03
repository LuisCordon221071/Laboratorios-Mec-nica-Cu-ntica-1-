import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def calcular_derivadas(t, estado):
    a1, a2 = estado
    derivada_a1 = -1j * (1.0 * a1 + 1.0 * a2)
    derivada_a2 = -1j * (1.0 * a1 + 2.0 * a2)
    return [derivada_a1, derivada_a2]

estado_inicial = [1.0 + 0.0j, 0.0 + 0.0j]
tiempo = [0, 10]
puntos_tiempo = np.linspace(tiempo[0], tiempo[1], 1000)

resultado = solve_ivp(calcular_derivadas, tiempo, estado_inicial, 
                     t_eval=puntos_tiempo, method='RK45')

probabilidad_1 = np.abs(resultado.y[0])**2
probabilidad_2 = np.abs(resultado.y[1])**2

plt.figure(figsize=(10, 6))
plt.plot(resultado.t, probabilidad_1, label='$|a_1(t)|^2$')
plt.plot(resultado.t, probabilidad_2, label='$|a_2(t)|^2$')
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.title('Sistema: $E_1=1$, $E_2=2$, $V=1$')
plt.legend()
plt.grid(True)
plt.show()

print("Norma inicial:", np.abs(estado_inicial[0])**2 + np.abs(estado_inicial[1])**2)
print("Norma final:", probabilidad_1[-1] + probabilidad_2[-1])