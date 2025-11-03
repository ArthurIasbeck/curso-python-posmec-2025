import numpy as np

# Lista comum do Python (em newtons)
forcas_lista = [120, 250, 400, 310, 500]

# Convertendo a lista para um array NumPy
forcas = np.array(forcas_lista)

# Converter para kN
forcas_kN = forcas / 1000

# Calcular a média e o valor máximo
media = np.mean(forcas)
maxima = np.max(forcas)

print(f"Forças em kN: {forcas_kN}")
print(f"Média = {media:.2f} N, Máxima = {maxima:.2f} N")
