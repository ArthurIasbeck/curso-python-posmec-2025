import numpy as np

# Vetor de forças medidas (N)
forca = np.array([120, 250, 310, 400, 280, 500, 260])

# Vetor de tempo correspondente (s)
tempo = np.linspace(0, 15, len(forca))

# Limite máximo permitido
limite = 300

# np.where retorna os índices onde a condição é verdadeira
indices = np.nonzero(forca > limite)

print("Índices onde F > 300 N:", indices)
print("Tempos correspondentes:", tempo[indices])
print("Forças correspondentes:", forca[indices])

forca = forca[tempo > 10]
tempo = tempo[tempo > 10]

print("\nApós filtrar por tempo > 10 s:")
print("Forças:", forca)
print("Tempos:", tempo)
