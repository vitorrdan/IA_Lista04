# decision_tree_lib/utils.py
# decision_tree_lib/utils.py

import numpy as np
import pandas as pd

def calculate_entropy(y: pd.Series) -> float:
    """
    Calcula a entropia de um conjunto de rótulos. 
    
    A entropia mede a impureza ou desordem de um conjunto de dados.
    Fórmula: Entropy(S) = sum(-p_i * log2(p_i))
    
    Args:
        y: Uma Series do pandas contendo os rótulos da classe (ex: df['play']).
        
    Returns:
        O valor da entropia como um float.
    """
    # Conta a ocorrência de cada classe
    class_counts = y.value_counts()
    
    # Calcula a probabilidade de cada classe
    probabilities = class_counts / len(y)
    
    # Calcula a entropia
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_gini_index(y: pd.Series) -> float:
    """
    [cite_start]Calcula o índice de Gini de um conjunto de rótulos. [cite: 51]

    O índice Gini também mede a impureza de um conjunto de dados.
    Fórmula: Gini(S) = 1 - sum(p_i^2)

    Args:
        y: Uma Series do pandas contendo os rótulos da classe.

    Returns:
        O valor do índice de Gini como um float.
    """
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    
    # Calcula o Gini
    gini = 1 - np.sum(probabilities**2)
    
    return gini

def calculate_information_gain(data: pd.DataFrame, attribute_name: str, target_name: str) -> float:
    """
    [cite_start]Calcula o ganho de informação de um atributo. [cite: 51]
    [cite_start]É a métrica usada pelo algoritmo ID3. [cite: 9]
    
    Args:
        data: O DataFrame completo.
        attribute_name: O nome do atributo (coluna) para calcular o ganho.
        target_name: O nome da coluna alvo.
        
    Returns:
        O ganho de informação como um float.
    """
    # 1. Calcular a entropia total do conjunto de dados
    total_entropy = calculate_entropy(data[target_name])
    
    # 2. Calcular a entropia ponderada para o atributo
    weighted_entropy = 0
    unique_values = data[attribute_name].unique()
    
    for value in unique_values:
        subset = data[data[attribute_name] == value]
        weight = len(subset) / len(data)
        subset_entropy = calculate_entropy(subset[target_name])
        weighted_entropy += weight * subset_entropy
        
    # 3. Retornar a diferença
    information_gain = total_entropy - weighted_entropy
    return information_gain

def calculate_gain_ratio(data: pd.DataFrame, attribute_name: str, target_name: str) -> float:
    """
    [cite_start]Calcula a razão de ganho de um atributo. [cite: 51]
    [cite_start]É a métrica usada pelo algoritmo C4.5. [cite: 10]

    Args:
        data: O DataFrame completo.
        attribute_name: O nome do atributo (coluna) para calcular a razão de ganho.
        target_name: O nome da coluna alvo.
        
    Returns:
        A razão de ganho como um float.
    """
    # 1. Calcular o Ganho de Informação
    information_gain = calculate_information_gain(data, attribute_name, target_name)
    
    # 2. Calcular o Split Info do atributo
    attribute_counts = data[attribute_name].value_counts()
    attribute_probabilities = attribute_counts / len(data)
    split_info = -np.sum(attribute_probabilities * np.log2(attribute_probabilities))
    
    # 3. Evitar divisão por zero
    if split_info == 0:
        return 0
        
    # 4. Retornar a razão
    gain_ratio = information_gain / split_info
    return gain_ratio