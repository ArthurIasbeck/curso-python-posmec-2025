import numpy as np

# Lista comum do Python (em newtons)
forcas_lista = [120, 250, 400, 310, 500]

# Convertendo a lista para um array NumPy
forcas = np.array(forcas_lista)

print("Tipo do objeto:", type(forcas))
print("Vetor de forças (N):", forcas)
