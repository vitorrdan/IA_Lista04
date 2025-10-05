import pandas as pd
from . import utils  # Usando import relativo para chamar funcoes do utils.py

class ID3:
    """
    Uma implementação do zero do algoritmo de árvore de decisão ID3.
    
    Atributos:
        tree_ (dict): A estrutura da árvore de decisão aprendida, representada
                     como um dicionário aninhado.
    """
    def __init__(self):
        self.tree_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Constrói a árvore de decisão a partir do conjunto de treinamento (X, y).

        Args:
            X: DataFrame contendo os atributos de treinamento (discretizados).
            y: Series contendo o atributo alvo.
        """
        # Combina X e y em um único DataFrame para facilitar a manipulação
        data = X.copy()
        data[y.name] = y
        
        # Inicia a construção recursiva da árvore
        self.tree_ = self._build_tree(data, data, X.columns.tolist(), y.name)


    def _build_tree(self, data: pd.DataFrame, original_data: pd.DataFrame, feature_names: list, target_name: str):
        """
        Função recursiva que constrói a árvore de decisão.
        
        """
        # --- CASOS BASE (CONDIÇÕES DE PARADA) ---

        # 1. Se todos os exemplos restantes pertencem à mesma classe
        if len(data[target_name].unique()) == 1:
            return data[target_name].unique()[0]

        # 2. Se não há mais atributos para dividir ou não há mais dados
        if len(feature_names) == 0 or len(data) == 0:
            # Retorna a classe majoritária dos dados originais (do nó pai)
            majority_class = original_data[target_name].mode()[0]
            return majority_class

        # --- PASSO RECURSIVO ---

        # 3. Encontrar o melhor atributo para dividir usando Ganho de Informação
        gains = [utils.calculate_information_gain(data, feature, target_name) for feature in feature_names]
        best_feature_index = gains.index(max(gains))
        best_feature = feature_names[best_feature_index]
        
        # Estrutura da árvore: um dicionário onde a chave é o melhor atributo
        tree = {best_feature: {}}
        
        # Remover o atributo escolhido da lista para as próximas recursões
        remaining_features = [f for f in feature_names if f != best_feature]
        
        # 4. Criar um galho para cada valor possível do melhor atributo
        for value in data[best_feature].unique():
            # Filtrar o subconjunto de dados para o valor atual
            subset = data[data[best_feature] == value]
            
            # Chamada recursiva para construir a sub-árvore
            subtree = self._build_tree(subset, data, remaining_features, target_name)
            
            # Adicionar a sub-árvore ao galho correspondente
            tree[best_feature][value] = subtree
            
        return tree
    
    
    # capacidade de predição
    def predict(self, X: pd.DataFrame) -> list:
        """
        Faz previsões para um conjunto de dados X.
        """
        # Converte o DataFrame para uma lista de dicionários (um para cada linha)
        rows = X.to_dict(orient='records')
        predictions = []
        for row in rows:
            predictions.append(self._predict_single(row, self.tree_))
        return predictions

    def _predict_single(self, row: dict, tree: dict):
        """
        Função auxiliar recursiva para classificar uma única instância (linha).
        """
        # Se o nó atual não for um dicionário, é uma folha (predição final)
        if not isinstance(tree, dict):
            return tree

        # Pega o atributo do nó atual
        attribute = next(iter(tree))
        
        # Pega o valor desse atributo na linha de dados
        value = row[attribute]
        
        # Se o valor não foi visto no treino, pode dar erro. 
        # Uma implementação mais robusta teria um fallback aqui.
        if value not in tree[attribute]:
            # Fallback simples: retorna a primeira folha que encontrar (pode ser melhorado)
            return list(tree[attribute].values())[0]

        # Navega para o próximo nó (sub-árvore)
        next_node = tree[attribute][value]
        
        # Chama a função recursivamente
        return self._predict_single(row, next_node)
