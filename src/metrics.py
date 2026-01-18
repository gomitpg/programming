import numpy as np

def classification_metrics(y_true, y_pred):
    """
    Calcula as métricas de classificação (accuracy global e recall por classe)

    Parâmetros:
    y_true: vetor com labels reais
    y_pred: vetor com a previsão das labels
    """

    # confirmar o comprimento das variáveis
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred têm que ter a mesma dimensão")

    total = len(y_true)
    correct = 0

    class_counts = {}
    class_correct = {}

    
    for i in range(total):
        true_label = y_true[i]
        pred_label = y_pred[i]

        # accuracy global
        if true_label == pred_label:
            correct += 1

        
        if true_label not in class_counts:
            class_counts[true_label] = 0
            class_correct[true_label] = 0

        class_counts[true_label] += 1

        if pred_label == true_label:
            class_correct[true_label] += 1

    # accuracy por classe
    per_class_accuracy = {}

    for cls in class_counts:
        per_class_accuracy[cls] = class_correct[cls] / class_counts[cls]

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy
    }