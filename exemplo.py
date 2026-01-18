import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # apenas para ir buscar o dataset Iris

from src.knn import knn_predict
from src.metrics import classification_metrics

def main():

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
    metric = 'euclidean'
    normalize_data = 'standardize'

    #Train/Test Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_ratio = 0.2
    n_test = int(len(X) * test_ratio)

    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test  = X[test_idx]
    y_test  = y[test_idx]


    # Correr o algoritmo:
    y_pred = knn_predict(   # correr KNN
        X_train=X_train,          # X train vai treinar com base nas 4 features
        y_train=y_train,            
        X_test=X_test,          # X test vai testar com base nas 4 features
        k=k,
        metric = metric,
        normalize_data = normalize_data
    )

    #métricas
    accuracy=classification_metrics(y_test,y_pred)
    accuracy_global=accuracy['accuracy']
    accuracy_class=accuracy['per_class_accuracy']
    accuracy_class_1=accuracy_class[0]
    accuracy_class_2=accuracy_class[1]
    accuracy_class_3=accuracy_class[2]

    print(f'A Accuracy global é {accuracy_global:.02%}')
    print(f'A Accuracy da classe 1 {accuracy_class_1:.02%}')
    print(f'A Accuracy da classe 2 {accuracy_class_2:.02%}')
    print(f'A Accuracy da classe 3 {accuracy_class_3:.02%}')


    # Print na consola para ver se correu bem
    print("O script correu com sucesso")

    # Extrair as primeiras duas features para fazer plot 2D
    X_test_2d = X_test[:, :2]
    correct = (y_pred == y_test)

    edge_colors = ['black' if ok else 'red' for ok in correct]
    sizes = [60 if ok else 120 for ok in correct]  # opcional, mas recomendado

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_test_2d[:, 0],
        X_test_2d[:, 1],
        c=y_pred,
        cmap="viridis",
        edgecolors=edge_colors,     # <-- plural
        linewidths=1.5,
        s=sizes
    )

    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
    plt.title(f"KNN classification (k={k}, metric={metric}) | Accuracy={accuracy_global:.2%}")
    plt.savefig("knn_plot.png", dpi=150)
    plt.show()
    

if __name__ == "__main__":
    main()