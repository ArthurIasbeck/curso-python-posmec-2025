import numpy as np

# Lista comum do Python (em newtons)
forcas_lista = [120, 250, 400, 310, 500]

# Convertendo a lista para um array NumPy
forcas = np.array(forcas_lista)

print(f"2 * forcas_lista = {2 * forcas_lista}")
print(f"2 * forcas = {2 * forcas}")
