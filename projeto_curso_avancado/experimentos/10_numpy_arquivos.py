import numpy as np

# Geração de dados simulados
t = np.linspace(0, 5, 10)  # tempo [s]
x = 0.02 * np.sin(2 * np.pi * 3 * t)  # deslocamento [m] (amplitude 2 cm, freq. 3 Hz)

print("Dados gerados:")
print(f"tempo = {t}")
print(f"deslocamento = {x}")

# Empacotar dados em uma única matriz (colunas)
dados = np.column_stack((t, x))

# Salvando os dados em arquivo TXT
np.savetxt(
    "vibracao.txt",
    dados,
    fmt="%.6f",
    delimiter="\t",
    header="tempo(s)\tdeslocamento(m)",
)

print("Arquivo 'vibracao.txt' salvo com sucesso!")

# Carregando os dados do arquivo
dados_carregados = np.loadtxt("vibracao.txt", delimiter="\t", skiprows=1)

# Separando colunas
tempo = dados_carregados[:, 0]
desloc = dados_carregados[:, 1]

print("Dados carregados do arquivo:")
print(f"tempo = {tempo}")
print(f"deslocamento = {desloc}")
