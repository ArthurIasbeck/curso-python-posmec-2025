import pandas as pd

dados = {
    "Deformação (%)": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    "Tensão (MPa)": [0, 50, 100, 150, 200, 230],
}

df = pd.DataFrame(dados)

print("DataFrame original")
print(df)

print("\n\n\nInformações gerais")
df.info()

print("\n\n\nEstatísticas descritivas")
print(df.describe())

print("\n\n\nPrimeiras linhas:")
print(df.head(3))

print("\n\n\nÚltimas linhas:")
print(df.tail(2))

print("\n\n\nApenas a coluna de tensão:")
print(df["Tensão (MPa)"])

# Adição de uma nova coluna
df["Deformação (decimal)"] = df["Deformação (%)"] / 100

# Calculando o módulo de elasticidade aproximado (E = tensão / deformação)
df["Módulo (MPa)"] = df["Tensão (MPa)"] / df["Deformação (decimal)"]

print("\n\n\nDataFrame atualizado")
print(df)

print("\n\n\nPontos onde a tensão é maior que 100 MPa")
print(df[df["Tensão (MPa)"] > 100])

df.to_excel("ensaio_tracao.xlsx", index=False)
print("\n\n\nArquivo 'ensaio_tracao.xlsx' salvo com sucesso!")

df_load = pd.read_excel("ensaio_tracao.xlsx")
print("\n\n\nArquivo recarregado:")
print(df_load)
