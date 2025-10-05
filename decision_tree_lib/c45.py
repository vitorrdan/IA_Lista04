import pandas as pd
import numpy as np
from . import utils

class C45:
    """
    Melhorias sobre o ID3:
    1. Usa Razão de Ganho (Gain Ratio) como critério de divisão.
    2. Lida nativamente com atributos contínuos.
    """
    def __init__(self):
        self.tree_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Constrói a árvore de decisão a partir do conjunto de treinamento (X, y).
        """
        data = X.copy()
        data[y.name] = y
        self.tree_ = self._build_tree(data, X.columns.tolist(), y.name)

    def _find_best_split(self, data: pd.DataFrame, feature_names: list, target_name: str):
        """
        Encontra o melhor atributo e/ou limiar para dividir os dados.
        Itera sobre todos os atributos e calcula a Razão de Ganho para cada um.
        """
        best_feature = None
        max_gain_ratio = -1
        best_threshold = None # Apenas para atributos contínuos

        for feature in feature_names:
            # VERIFICA SE O ATRIBUTO É CONTÍNUO OU CATEGÓRICO
            if pd.api.types.is_numeric_dtype(data[feature]):
                # Lógica para atributos contínuos
                threshold, info_gain = utils.find_best_continuous_split(data, feature, target_name)
                
                if threshold is None:
                    continue

                # C4.5 ainda usa Razão de Ganho, então precisamos calcular o Split Info para a divisão binária
                subset_le = data[data[feature] <= threshold]
                subset_gt = data[data[feature] > threshold]
                p_le = len(subset_le) / len(data)
                p_gt = len(subset_gt) / len(data)
                
                # Evita log(0)
                if p_le == 0 or p_gt == 0:
                    split_info = 0
                else:
                    split_info = - (p_le * np.log2(p_le) + p_gt * np.log2(p_gt))
                
                if split_info == 0:
                    gain_ratio = 0
                else:
                    gain_ratio = info_gain / split_info

                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold
            else:
                # Lógica para atributos categóricos
                gain_ratio = utils.calculate_gain_ratio(data, feature, target_name)
                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = None # Reseta para garantir que não usemos um limiar antigo

        return best_feature, best_threshold

    def _build_tree(self, data: pd.DataFrame, feature_names: list, target_name: str):
        """
        Função recursiva que constrói a árvore de decisão.
        """
        # --- CASOS BASE ---
        if len(data[target_name].unique()) == 1:
            return data[target_name].unique()[0]

        if len(feature_names) == 0 or len(data) < 2: # Adicionado len(data) < 2
            return data[target_name].mode()[0]

        # --- PASSO RECURSIVO ---
        best_feature, best_threshold = self._find_best_split(data, feature_names, target_name)

        if best_feature is None:
            return data[target_name].mode()[0]
        
        # Se o split for contínuo, a chave da árvore será diferente
        if best_threshold is not None:
            tree_key = f"{best_feature} <= {best_threshold:.2f}"
            tree = {tree_key: {}}
            
            # Divisão binária
            left_subset = data[data[best_feature] <= best_threshold]
            right_subset = data[data[best_feature] > best_threshold]
            
            tree[tree_key]['True'] = self._build_tree(left_subset, feature_names, target_name)
            tree[tree_key]['False'] = self._build_tree(right_subset, feature_names, target_name)
        else:
            # Lógica original do ID3 para atributos categóricos
            tree = {best_feature: {}}
            remaining_features = [f for f in feature_names if f != best_feature]
            
            for value in data[best_feature].unique():
                subset = data[data[best_feature] == value]
                tree[best_feature][value] = self._build_tree(subset, remaining_features, target_name)
                
        return tree
    
    
    # capacidade de predição
    def predict(self, X: pd.DataFrame) -> list:
        rows = X.to_dict(orient='records')
        return [self._predict_single(row, self.tree_) for row in rows]

    def _predict_single(self, row: dict, tree: dict):
        if not isinstance(tree, dict):
            return tree

        question = next(iter(tree))
        
        # Verifica se é uma pergunta de split contínuo
        if "<=" in question:
            feature, threshold = question.split(' <= ')
            threshold = float(threshold)
            
            if row[feature] <= threshold:
                branch = 'True'
            else:
                branch = 'False'
            
            return self._predict_single(row, tree[question][branch])
        else: # É um split categórico
            attribute = question
            value = row[attribute]
            if value not in tree[attribute]:
                return list(tree[attribute].values())[0][0] # Fallback mais complexo
            
            next_node = tree[attribute][value]
            return self._predict_single(row, next_node)