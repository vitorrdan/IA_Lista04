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
    Calcula o índice de Gini de um conjunto de rótulos. 

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
    Calcula o ganho de informação de um atributo. 
    É a métrica usada pelo algoritmo ID3. 
    
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
    Calcula a razão de ganho de um atributo. 
    É a métrica usada pelo algoritmo C4.5. 

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

def find_best_continuous_split(data: pd.DataFrame, attribute_name: str, target_name: str) -> tuple:
    """
    Encontra o melhor limiar para dividir um atributo CONTÍNUO.
    Implementa a "varredura por limiar" testando os pontos médios.

    Args:
        data: DataFrame com os dados.
        attribute_name: Nome do atributo contínuo.
        target_name: Nome do atributo alvo.

    Returns:
        Uma tupla contendo (melhor_limiar, maior_ganho_de_informacao).
    """
    total_entropy = calculate_entropy(data[target_name])
    unique_values = sorted(data[attribute_name].unique())
    best_threshold = None
    max_info_gain = -1

    if len(unique_values) < 2:
        return None, -1

    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i+1]) / 2
        
        left_subset = data[data[attribute_name] <= threshold]
        right_subset = data[data[attribute_name] > threshold]

        # Ignorar divisões que não separam os dados
        if len(left_subset) == 0 or len(right_subset) == 0:
            continue
            
        # Calcular entropia ponderada da divisão binária
        weight_left = len(left_subset) / len(data)
        entropy_left = calculate_entropy(left_subset[target_name])
        
        weight_right = len(right_subset) / len(data)
        entropy_right = calculate_entropy(right_subset[target_name])
        
        weighted_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)
        
        # Calcular o ganho de informação para esta divisão
        current_info_gain = total_entropy - weighted_entropy
        
        if current_info_gain > max_info_gain:
            max_info_gain = current_info_gain
            best_threshold = threshold
            
    return best_threshold, max_info_gain


def find_best_continuous_split_gini(data: pd.DataFrame, attribute_name: str, target_name: str) -> tuple:
    """
    Encontra o melhor limiar para um atributo contínuo usando o Ganho Gini.
    """
    unique_values = sorted(data[attribute_name].unique())
    best_threshold = None
    max_gini_gain = -1

    if len(unique_values) < 2:
        return None, -1

    # Gini do nó pai
    parent_gini = calculate_gini_index(data[target_name])

    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i+1]) / 2
        
        left_subset = data[data[attribute_name] <= threshold]
        right_subset = data[data[attribute_name] > threshold]

        if len(left_subset) == 0 or len(right_subset) == 0:
            continue
            
        # Gini ponderado dos filhos
        gini_left = calculate_gini_index(left_subset[target_name])
        gini_right = calculate_gini_index(right_subset[target_name])
        
        weight_left = len(left_subset) / len(data)
        weight_right = len(right_subset) / len(data)
        
        weighted_gini_children = (weight_left * gini_left) + (weight_right * gini_right)
        
        # Ganho Gini = Gini(pai) - Gini_ponderado(filhos)
        current_gini_gain = parent_gini - weighted_gini_children
        
        if current_gini_gain > max_gini_gain:
            max_gini_gain = current_gini_gain
            best_threshold = threshold
            
    return best_threshold, max_gini_gain