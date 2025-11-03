componentes = ["Rotor", "Eixo", "Acoplamento"]
massas = [12.0, 5.0, 3.5]
velocidades = [180, 220, 150]

energias = {
    componente: 0.5 * m * v**2
    for componente, m, v in zip(componentes, massas, velocidades)
}

print(energias)
