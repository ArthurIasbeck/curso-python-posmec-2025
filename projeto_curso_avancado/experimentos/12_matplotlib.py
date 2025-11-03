import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2, 500)  # Tempo [s]
x = 0.02 * np.sin(2 * np.pi * 3 * t)  # Deslocamento [m]
v = np.gradient(x, t)  # Velocidade [m/s]

plt.figure(figsize=(8, 4), dpi=200)  # Cria a figura com tamanho personalizado
plt.plot(t, x, label="Deslocamento")  # Curva de posição
plt.plot(t, v, label="Velocidade")  # Curva de velocidade
plt.title("Resposta harmônica de um sistema massa–mola")  # Define o título do gráfico
plt.xlabel("Tempo [s]")  # Define o rótulo do eixo x
plt.ylabel("Deslocamento [m]")  # Define o rótulo do eixo y
plt.grid()  # Insere a grade
plt.legend()  # Insere a legenda
plt.tight_layout()  # Ajusta o espaçamento
plt.savefig("resposta_harmonica.eps")  # Salva o gráfico como arquivo PNG
plt.show()  # Exibe o gráfico
