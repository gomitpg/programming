import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # apenas para ir buscar o dataset Iris

from src.knn import knn_predict

def main():
    """
    Docstring for main
    """

    # Definir parâmetros a utilizar:
    data = load_iris() # carregar dataset Iris
        # este dataset contém:
            # data.data -> matriz (n_samples, 4)
                # SepalLengthCm -> data.data [:,0]
                # SepalWidthCm -> data.data [:,1]
                # PetalLengthCm -> data.data [:,2]
                # PetalWidthCm -> data.data [:,3]
            # Species -> y.target
    X = data.data
    y = data.target
    k = 5 # define o k aqui em vez de ser na chamada da função, assim altera logo nos visuais gerados
    metric = 'manhattan'
    normalize_data = 'standardize'

    # Correr o algoritmo:
    y_pred = knn_predict(   # correr KNN
        X_train=X,          # X train vai treinar com base nas 4 features
        y_train=y,            
        X_test=X,           # X test vai testar com base nas 4 features
        k=k,
        metric = metric,
        normalize_data = normalize_data
    )

    # Print na consola para ver se correu bem
    print("O script correu com sucesso")

    # Extrair as primeiras duas features para fazer plot 2D
    X_2d = X[:, :2]

    # visualização gráfica
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_pred,
        cmap="viridis",
        edgecolor="k"
    )
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
    plt.title(f"KNN classification on Iris dataset (k={k}, metric={metric})")
    plt.savefig("knn_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()