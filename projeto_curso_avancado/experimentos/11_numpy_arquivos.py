import numpy as np

# Geração de dados (simulação simples)
t = np.linspace(0, 5, 1000)  # tempo [s]
x = 0.02 * np.exp(-0.2 * t) * np.cos(2 * np.pi * 3 * t)  # deslocamento [m]
v = np.gradient(x, t)  # velocidade [m/s]
a = np.gradient(v, t)  # aceleração [m/s²]

# Salvando tudo em um único arquivo .npz
np.savez("simulacao_vibracao.npz", tempo=t, deslocamento=x, velocidade=v, aceleracao=a)

print("Arquivo 'simulacao_vibracao.npz' salvo com sucesso!")

# Carregando o arquivo salvo
dados = np.load("simulacao_vibracao.npz")

# Acessando os arrays pelo nome
t2 = dados["tempo"]
x2 = dados["deslocamento"]
v2 = dados["velocidade"]
a2 = dados["aceleracao"]

# Verificando
print("\nChaves armazenadas:", list(dados.keys()))
print("Forma do vetor tempo:", t2.shape)
print("Primeiros valores de deslocamento:", x2[:5])
