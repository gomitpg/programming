import numpy as np
from src.distance import distance
from src.scaler import scaler

def knn_predict(X_train, y_train, X_test, k=3, metric = 'euclidian', normalize_data=False):
    """
    Metodo de classificacao supervisionada K nearest neighbors
    
    Parâmetros:
    X_train: vetor n-dimensional com os dados de treino X
    y_train: vetor n-dimensional com os dados de treino y
    X_test: vetor n-dimensional com os dados de teste X
    k: numero de neighbors a considerar
    metric: euclidian, manhattan
    :param normalize_data: standardize, minmax
    """

    if normalize_data == 'standardize':
        X_train = scaler(X_train, method='standardize')
        X_test  = scaler(X_test,  method='standardize')

    elif normalize_data == 'minmax':
        X_train = scaler(X_train, method='minmax')
        X_test  = scaler(X_test,  method='minmax')

    predictions = []

    for x_test in X_test:
        distances = []

        for i in range(len(X_train)):
            x_train = X_train[i]
            y_label = y_train[i]

            dist = distance(x_test, x_train, metric = metric) # chama a funcao dist, passando o parâmetro metric
            distances.append((dist, y_label))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]

        # contar labels com dicionário
        label_counts = {}

        for dist, label in k_nearest:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # encontrar a label com mais contagem
        max_count = -1
        prediction = None

        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                prediction = label
       

        predictions.append(prediction)

    return np.array(predictions)