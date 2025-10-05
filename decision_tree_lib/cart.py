import pandas as pd
from itertools import combinations
from . import utils
import re

class CART:
    """
    Uma implementação do zero do algoritmo CART para classificação.
    
    Características:
    1. Usa o Índice Gini como critério de pureza.
    2. Realiza divisões estritamente binárias.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Constrói a árvore de decisão a partir do conjunto de treinamento."""
        data = X.copy()
        data[y.name] = y
        self.tree_ = self._build_tree(data, y.name, depth=0)
        
      # --- MÉTODO PÚBLICO DE PREDIÇÃO (ESTAVA FALTANDO) ---
    def predict(self, X: pd.DataFrame) -> list:
        """
        Faz previsões para um conjunto de dados X.
        """
        rows = X.to_dict(orient='records')
        return [self._predict_single(row, self.tree_) for row in rows]    

    def _calculate_gini_gain(self, data, split_subsets, target_name):
        """Calcula o Ganho Gini para uma dada divisão binária."""
        parent_gini = utils.calculate_gini_index(data[target_name])
        
        left_subset, right_subset = split_subsets
        
        if len(left_subset) == 0 or len(right_subset) == 0:
            return 0
            
        gini_left = utils.calculate_gini_index(left_subset[target_name])
        gini_right = utils.calculate_gini_index(right_subset[target_name])
        
        weight_left = len(left_subset) / len(data)
        weight_right = len(right_subset) / len(data)
        
        weighted_gini_children = (weight_left * gini_left) + (weight_right * gini_right)
        
        return parent_gini - weighted_gini_children

    def _find_best_split(self, data: pd.DataFrame, feature_names: list, target_name: str):
        """Encontra a melhor divisão binária possível para os dados."""
        best_feature, max_gini_gain, best_split_value = None, -1, None

        for feature in feature_names:
            unique_values = data[feature].unique()
            
            if pd.api.types.is_numeric_dtype(data[feature]):
                # Lógica para atributos contínuos
                threshold, gini_gain = utils.find_best_continuous_split_gini(data, feature, target_name)
                if gini_gain > max_gini_gain:
                    max_gini_gain, best_feature, best_split_value = gini_gain, feature, threshold
            else:
                # Lógica para atributos categóricos: testar todas as partições binárias
                if len(unique_values) < 2: continue
                
                # Gera todas as combinações de subconjuntos
                for i in range(1, len(unique_values) // 2 + 1):
                    for subset in combinations(unique_values, i):
                        left_values = set(subset)
                        
                        left_subset = data[data[feature].isin(left_values)]
                        right_subset = data[~data[feature].isin(left_values)]
                        
                        gini_gain = self._calculate_gini_gain(data, (left_subset, right_subset), target_name)
                        
                        if gini_gain > max_gini_gain:
                            max_gini_gain, best_feature, best_split_value = gini_gain, feature, left_values
                            
        return best_feature, best_split_value

    def _build_tree(self, data: pd.DataFrame, target_name: str, depth: int):
        """Função recursiva que constrói a árvore."""
        feature_names = [col for col in data.columns if col != target_name]
        
        # --- CASOS BASE ---
        if (len(data[target_name].unique()) == 1 or 
            len(data) < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)):
            return data[target_name].mode()[0]

        best_feature, best_split_value = self._find_best_split(data, feature_names, target_name)

        if best_feature is None:
            return data[target_name].mode()[0]
        
        # --- PASSO RECURSIVO ---
        if pd.api.types.is_numeric_dtype(data[best_feature]):
            tree_key = f"{best_feature} <= {best_split_value:.2f}"
            left_subset = data[data[best_feature] <= best_split_value]
            right_subset = data[data[best_feature] > best_split_value]
        else: # Categórico
            tree_key = f"{best_feature} in {best_split_value}"
            left_subset = data[data[best_feature].isin(best_split_value)]
            right_subset = data[~data[best_feature].isin(best_split_value)]

        tree = {tree_key: {}}
        tree[tree_key]['True'] = self._build_tree(left_subset, target_name, depth + 1)
        tree[tree_key]['False'] = self._build_tree(right_subset, target_name, depth + 1)

        return tree
    
    # capacidade de predição
    def _predict_single(self, row: dict, tree: dict):
        # Caso base: se não for um dicionário, é uma folha (predição).
        if not isinstance(tree, dict):
            return tree

        # Pega a pergunta do nó atual. Ex: "Sex <= 0.50"
        question = next(iter(tree))

        # Teste para split contínuo
        if "<=" in question:
            feature, threshold = question.split(' <= ')
            threshold = float(threshold)
            branch = 'True' if row[feature] <= threshold else 'False'
        
        # Teste para split categórico
        elif "in" in question:
            match = re.match(r"(\w+)\s+in\s+({.*})", question)
            feature = match.group(1)
            value_set = eval(match.group(2)) 
            branch = 'True' if row[feature] in value_set else 'False'
        
        
        # Navega para o próximo nó (sub-árvore) usando o galho correto ('True' ou 'False')
        return self._predict_single(row, tree[question][branch])