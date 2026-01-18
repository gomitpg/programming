def scaler(vetor,method='standardize'):
    """
    Normaliza os dados feature a feature (coluna a coluna) através de min-max scaling (intervalo [0, 1]) ou normalização z-score.

    Parâmetros:
        vetor : numpy.ndarray
        method : standardize , minmax
 
    Devolve:
        numpy.ndarray
        Vetor normalizado
    """
     
    if method == 'standardize':
        vetor_mean = vetor.mean(axis=0)
        vetor_stdev = vetor.std(axis=0)
        for i in range(len(vetor_stdev)):
            if vetor_stdev[i] == 0:
                vetor_stdev[i]=1
        return (vetor - vetor_mean) / vetor_stdev
    
    elif method == 'minmax':
        vetor_min = vetor.min(axis=0)
        vetor_max = vetor.max(axis=0)
        for i in range (len(vetor_min)):
            if vetor_max[i]== vetor_min[i]:
                vetor_max[i]=vetor_max[i]+1
        return (vetor - vetor_min) / (vetor_max - vetor_min)
    
    else:
        raise ValueError('Escolhe um dos scalers disponíveis (standardize ou minmax)')