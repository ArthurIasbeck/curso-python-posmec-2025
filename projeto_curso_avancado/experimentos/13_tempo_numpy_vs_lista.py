import numpy as np
import time

N = 10_000_000  # número de elementos (10 milhões)
rng = np.random.default_rng(0)

# Massa e velocidade
m = 2.5  # kg
velocidades_lista = list(rng.random(N))  # lista Python
velocidades_np = rng.random(N)  # array NumPy

# Cálculo da energia cinética: Ec = 0.5 * m * v²
# Utilizando listas
tic = time.time()
energias_lista = []
for v in velocidades_lista:
    energias_lista.append(0.5 * m * v**2)
toc = time.time()
tempo_lista = toc - tic

# Utilizando Numpy
tic = time.time()
energias_np = 0.5 * m * velocidades_np**2
toc = time.time()
tempo_numpy = toc - tic

print(f"Tempo com LISTAS: {tempo_lista:.4f} s")
print(f"Tempo com NUMPY:  {tempo_numpy:.4f} s")
print(f"NumPy é aproximadamente {tempo_lista/tempo_numpy:.0f}x mais rápido!")
