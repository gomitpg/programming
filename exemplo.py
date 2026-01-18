import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # apenas para ir buscar o dataset Iris

from src.knn import knn_predict

def main():
    data = load_iris() # carregar dataset Iris
    X = data.data
    y = data.target

    # usar apenas duas features para visualização
    X_2d = X[:, :2]

    # aplicar KNN
    y_pred = knn_predict(
        X_train=X_2d,
        y_train=y,
        X_test=X_2d,
        k=5,
        normalize_data="standardize"
    )

    # output simples no terminal
    print("O script correu com sucesso")

    # visualização gráfica
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_pred,
        cmap="viridis",
        edgecolor="k"
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("KNN classification on Iris dataset (k=5)")
    plt.savefig("knn_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()