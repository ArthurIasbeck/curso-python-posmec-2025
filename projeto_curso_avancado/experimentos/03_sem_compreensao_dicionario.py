componentes = ["Rotor", "Eixo", "Acoplamento"]
massas = [12.0, 5.0, 3.5]  # kg
velocidades = [180, 220, 150]  # m/s

energias = {}  # dicionário vazio
for componente, m, v in zip(componentes, massas, velocidades):
    E = 0.5 * m * v**2
    energias[componente] = E

print(energias)
