import joblib
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ModeloRna:
    def __init__(self):

        # Variáveis de configuração
        self.seed = 0
        self.arquivo_dados = "input/dados.txt"
        self.nome_modelo = "output/modelo_rna.keras"
        self.arquivo_metricas = "output/metricas.npz"
        self.arquivo_scaller_x = "output/x_scaler.save"
        self.arquivo_scaller_y = "output/y_scaler.save"
        self.neuronios_camadas_ocultas = (64, 32)
        self.horizonte_predicao = 3
        self.epochs = 50
        self.batch_size = 64

        # Definição da semente aleatória para reprodutibilidade
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Conjunto de dados para treinamento
        self.t = None
        self.s = None
        self.i = None

        # Dataset completo
        self.X_in = None
        self.y = None

        # Dataset de treinamento
        self.y_train = None
        self.X_train = None

        # Dataset de validação
        self.y_val = None
        self.X_val = None

        # Dataset de teste
        self.y_test = None
        self.X_test = None

    def carregar_dados(self, plot_data=True):
        dados = np.loadtxt(self.arquivo_dados, skiprows=1)
        self.t = dados[:, 0]
        self.i = dados[:, 1]
        self.s = dados[:, 2]

        if plot_data:
            plt.figure(figsize=(10, 5), dpi=150)

            plt.subplot(2, 1, 1)
            plt.plot(self.t, self.i)
            plt.grid()
            plt.ylabel("Current (A)")

            plt.subplot(2, 1, 2)
            plt.plot(self.t, self.s)
            plt.grid()
            plt.xlabel("Time (s)")
            plt.ylabel("Displacement ($\mu$m)")

            plt.savefig("output/dados_treino.png")

    def construir_dataset(self):
        self.carregar_dados(plot_data=False)
        k_max = len(self.t) - 1

        X = []
        y = []
        k_vals = []
        for k in range(self.horizonte_predicao, k_max - self.horizonte_predicao):
            X.append(
                [
                    *self.i[k - self.horizonte_predicao : k],
                    *self.s[k - self.horizonte_predicao : k],
                ]
            )
            y.append((self.s[k : k + self.horizonte_predicao]))
            k_vals.append(k)

        self.X_in = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)
        k_vals = np.array(k_vals, dtype=int)

        # Dimensão dos conjuntos de treinamento, validação e teste
        n = len(self.X_in)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        # Dados de treinamento
        self.X_train = self.X_in[:n_train]
        self.y_train = self.y[:n_train]
        k_train = k_vals[:n_train]

        # Dados de validação
        self.X_val = self.X_in[n_train : n_train + n_val]
        self.y_val = self.y[n_train : n_train + n_val]

        # Dados de teste
        self.X_test = self.X_in[n_train + n_val :]
        self.y_test = self.y[n_train + n_val :]

        # Gera arquivo Excel do dataset de treinamento
        output_path = "output/dataset_treino.xlsx"
        if not os.path.exists(output_path):
            cols_i = [
                f"i[k-{h}]" for h in range(self.horizonte_predicao, 0, -1)
            ]  # i[k-H], ..., i[k-1]
            cols_s_passado = [
                f"s[k-{h}]" for h in range(self.horizonte_predicao, 0, -1)
            ]  # s[k-H], ..., s[k-1]
            cols_s_futuro = [
                f"s[k+{h}]" for h in range(1, self.horizonte_predicao)
            ]  # s[k+0], ..., s[k+H-1]
            cols_s_futuro.insert(0, "s[k]")  # Adiciona s[k] no início

            # Concatena X_train e y_train para virar uma tabela única
            treino_concat = np.hstack(
                [k_train.reshape(-1, 1), self.X_train, self.y_train]
            )
            colunas = ["k"] + cols_i + cols_s_passado + cols_s_futuro

            df_treino = pd.DataFrame(treino_concat, columns=colunas)
            df_treino.to_excel(output_path, index=False)

    def treinar_modelo(self):
        loss = []
        mae = []
        val_loss = []
        val_mae = []
        historico_treinamento_carregado = False

        try:
            data = np.load(self.arquivo_metricas)
            loss = data["loss"]
            mae = data["mae"]
            val_loss = data["val_loss"]
            val_mae = data["val_mae"]
            historico_treinamento_carregado = True

        except Exception as ex:
            print(
                f"Não foi possível carregar métricas anteriores. Iniciando novo treinamento ({ex})."
            ),

        if not historico_treinamento_carregado:
            self.construir_dataset()

            # Scallers para normalização
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            x_scaler.fit(self.X_train)
            y_scaler.fit(self.y_train)

            joblib.dump(x_scaler, self.arquivo_scaller_x)
            joblib.dump(y_scaler, self.arquivo_scaller_y)

            # Normalização
            X_train_n = x_scaler.transform(self.X_train)
            X_val_n = x_scaler.transform(self.X_val)

            y_train_n = y_scaler.transform(self.y_train)
            y_val_n = y_scaler.transform(self.y_val)

            # Camada de entrada
            inputs = keras.Input(shape=(2 * self.horizonte_predicao,), name="features")
            x = layers.BatchNormalization()(inputs)
            x = layers.GaussianNoise(stddev=0.05)(x)

            # Camadas ocultas
            for neuronios_camada_oculta in self.neuronios_camadas_ocultas:
                x = layers.Dense(
                    neuronios_camada_oculta,
                    activation="relu",
                )(x)
                x = layers.BatchNormalization()(x)

            # Camada de saída
            outputs = layers.Dense(self.horizonte_predicao)(x)

            model = keras.Model(inputs, outputs)
            model.summary()

            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            history = model.fit(
                X_train_n,
                y_train_n,
                validation_data=(X_val_n, y_val_n),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=2,
            )

            learning_rate = np.array(history.history.get("learning_rate", []))
            loss = np.array(history.history["loss"])
            mae = np.array(history.history["mae"])
            val_loss = np.array(history.history["val_loss"])
            val_mae = np.array(history.history["val_mae"])

            np.savez(
                self.arquivo_metricas,
                learning_rate=learning_rate,
                loss=loss,
                mae=mae,
                val_loss=val_loss,
                val_mae=val_mae,
            )

            model.save(self.nome_modelo)

        plt.figure(figsize=(10, 5), dpi=180)
        plt.plot(loss, ".-", label="Loss (Treino)")
        plt.plot(mae, ".-", label="MAE")
        plt.plot(val_loss, ".-", label="Loss (Validação)")
        plt.plot(val_mae, ".-", label="MAE (Validação)")
        plt.xlabel("Época")
        plt.ylabel("Valor")
        plt.title("Treino/Validação")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/resultado_treinamnto.png")

    def validar_modelo(self, passo_horizonte=1, k_max=None):
        self.construir_dataset()

        x_scaler = load(self.arquivo_scaller_x)
        y_scaler = load(self.arquivo_scaller_y)

        # Normaliza conjuntos
        X_test_n = x_scaler.transform(self.X_test)
        y_test_n = y_scaler.transform(self.y_test)

        # Carrega modelo e avalia
        model = keras.models.load_model(self.nome_modelo)
        model.evaluate(X_test_n, y_test_n, batch_size=self.batch_size, verbose=2)

        # Número de amostras a exibir no gráfico
        k_max = k_max if k_max is not None else len(self.X_test)

        # Predição de s
        y_pred_n = model.predict(X_test_n[:k_max], verbose=0)
        s_pred = y_scaler.inverse_transform(y_pred_n)  # shape: (k_max, h)

        s_real = self.y_test  # shape: (k_max, h)

        fit = 1 - np.linalg.norm(
            s_pred[:k_max, passo_horizonte - 1] - s_real[:k_max, passo_horizonte - 1]
        ) / np.linalg.norm(
            s_real[:k_max, passo_horizonte - 1]
            - np.mean(s_real[:k_max, passo_horizonte - 1])
        )
        print("Fit (%): ", fit * 100)

        plt.figure(figsize=(10, 5), dpi=180)
        plt.plot(s_real[:k_max, passo_horizonte - 1], ".-", label="s real")
        plt.plot(s_pred[:k_max, passo_horizonte - 1], ".-", label="s predito")
        plt.xlabel("Amostra")
        plt.ylabel("Deslocamento ($\\mu$m)")
        plt.title(
            f"Validação do Modelo — s (passo {passo_horizonte} de {self.horizonte_predicao} do horizonte)"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/validacao_modelo.png")
