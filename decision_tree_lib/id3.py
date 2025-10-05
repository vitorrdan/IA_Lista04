# decision_tree_lib/id3.py (VERSÃO FINAL)

import pandas as pd
from . import utils
from collections import Counter

class ID3:
    def __init__(self):
        self.tree_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data[y.name] = y
        self.tree_ = self._build_tree(data, data, X.columns.tolist(), y.name)

    def _build_tree(self, data: pd.DataFrame, original_data: pd.DataFrame, feature_names: list, target_name: str):
        if len(data[target_name].unique()) == 1:
            return data[target_name].unique()[0]
        if len(feature_names) == 0 or len(data) == 0:
            return original_data[target_name].mode()[0]
        
        gains = [utils.calculate_information_gain(data, feature, target_name) for feature in feature_names]
        best_feature = feature_names[gains.index(max(gains))]
        
        tree = {best_feature: {}}
        remaining_features = [f for f in feature_names if f != best_feature]
        
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            subtree = self._build_tree(subset, data, remaining_features, target_name)
            tree[best_feature][value] = subtree
            
        return tree

    def predict(self, X: pd.DataFrame) -> list:
        rows = X.to_dict(orient='records')
        return [self._predict_single(row, self.tree_) for row in rows]

    def _get_all_leaves(self, tree_node):
        """Coleta todos os valores de folhas (predições) em uma sub-árvore."""
        if not isinstance(tree_node, dict):
            return [tree_node]
        
        leaves = []
        for branch in tree_node.values():
            leaves.extend(self._get_all_leaves(branch))
        return leaves

    def _get_majority_leaf(self, tree_node):
        """Retorna a folha mais comum (moda) de uma sub-árvore."""
        leaves = self._get_all_leaves(tree_node)
        if not leaves:
            return 0 # Fallback final, caso a árvore esteja vazia
        return Counter(leaves).most_common(1)[0][0]

    def _predict_single(self, row: dict, tree: dict):
        """Classifica uma única instância com o novo fallback."""
        if not isinstance(tree, dict):
            return tree

        attribute = next(iter(tree))
        value = row.get(attribute) # Usar .get() para evitar KeyError se a coluna não existir

        if value is None or value not in tree[attribute]:
            # Se o valor não for encontrado, retorna a classe majoritária da sub-árvore atual
            return self._get_majority_leaf(tree[attribute])

        next_node = tree[attribute][value]
        return self._predict_single(row, next_node)