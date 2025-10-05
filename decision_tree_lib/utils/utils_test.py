import pandas as pd
from utils import *

df_play_tennis = pd.read_csv("../../data/JogarTênis.csv")

# --- Início dos Testes ---

print("--- Testando com o dataset 'Play Tennis' ---")
print(f"Dataset com {len(df_play_tennis)} linhas.\n")

# Nomes da coluna alvo e do atributo a ser testado
target_column = 'play'
attribute_to_test = 'outlook'

# 1. Testando a Entropia e o Gini do dataset completo
total_entropy = calculate_entropy(df_play_tennis[target_column])
total_gini = calculate_gini_index(df_play_tennis[target_column])
print(f"Métricas de impureza do dataset completo ({target_column}):")
print(f"  - Entropia: {total_entropy:.4f}")
print(f"  - Índice Gini: {total_gini:.4f}\n")


# 2. Testando o Ganho de Informação (para o ID3)
ig_outlook = calculate_information_gain(df_play_tennis, attribute_to_test, target_column)
print(f"Métricas para o atributo '{attribute_to_test}':")
print(f"  - Ganho de Informação (ID3): {ig_outlook:.4f}")

# 3. Testando a Razão de Ganho (para o C4.5)
gr_outlook = calculate_gain_ratio(df_play_tennis, attribute_to_test, target_column)
print(f"  - Razão de Ganho (C4.5): {gr_outlook:.4f}\n")


# Verificando para outros atributos para comparação
ig_temp = calculate_information_gain(df_play_tennis, 'temperature', target_column)
ig_humidity = calculate_information_gain(df_play_tennis, 'humidity', target_column)
ig_windy = calculate_information_gain(df_play_tennis, 'windy', target_column)

print("--- Comparação do Ganho de Informação entre todos os atributos ---")
print(f"Outlook: {ig_outlook:.4f}")
print(f"Temp: {ig_temp:.4f}")
print(f"Humidity: {ig_humidity:.4f}")
print(f"Windy: {ig_windy:.4f}")
print("\nConclusão: 'Outlook' seria escolhido como o primeiro nó pelo ID3, pois tem o maior Ganho de Informação.")