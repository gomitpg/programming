import numpy as np

def distance(v1, v2, metric = 'euclidean'):
    """
    Computa a distância entre dois pontos num espaço N-dimensional.
    Os inputs têm de ser vetores com a mesma dimensão.

    Parâmetros:
    v1 : array numpy com coordenadas do primeiro ponto
    v2 : array numpy com coordenadas do segundo ponto
    metric : euclidiana, manhattan

    Devolve:
    dist: distância entre v1 e v2 (float)
    """        
    if v1.shape != v2.shape: # verifica se v1 e v2 têm a mesma dimensão
        raise ValueError("v1 and v2 têm de ter a mesma dimensão") # devolve erro
    else:
        if metric == 'euclidean':
            dist_euc = np.sqrt(np.sum((v1 - v2) ** 2)) # computa distância euclidiana, numpy faz o somatório da substração elemento a elemento e o quadrado também
            return dist_euc # devolve a distância euclidiana
        elif metric == 'manhattan':
            dist_manhattan = np.sum(np.abs(v1 - v2))
            return dist_manhattan
        else: 
            raise ValueError(f"Métrica de distância desconhecida: {metric}. Escolhe por favor uma das opções")
        
