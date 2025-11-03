import numpy as np

# Definição da matriz de rigidez (N/m)
K = np.matrix([[1000, -500], [-500, 1000]])

# Vetor de forças aplicadas (N)
F = np.matrix([[100], [50]])

print(f"Matriz K (rigidez):\n {K}")
print(f"\nVetor de forças F:\n {F}")

print(f"\nTransposta de K:\n {K.T}")
print(f"\nInversa de K (flexibilidade):\n {K.I}")

u = K.I * F
print(f"\nDeslocamentos (u = K⁻¹·F):\n {u}")
