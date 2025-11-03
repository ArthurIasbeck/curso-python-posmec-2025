from src.modelo_rna import ModeloRna
import matplotlib.pyplot as plt


def main():
    modelo_rna = ModeloRna()
    modelo_rna.construir_dataset()
    # modelo_rna.treinar_modelo()
    # modelo_rna.validar_modelo(passo_horizonte=1, k_max=500)


if __name__ == "__main__":
    main()
    plt.show()
