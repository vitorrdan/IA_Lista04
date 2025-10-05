# decision_tree_lib/preprocessing.py
import pandas as pd

def clean_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a limpeza de dados específica para o Titanic.
    - Preenche valores nulos de 'Age' e 'Embarked'.
    - Converte 'Sex' e 'Embarked' para valores numéricos.
    - Remove colunas não utilizadas.

    Args:
        df: DataFrame do Titanic.

    Returns:
        DataFrame limpo e pronto para os algoritmos.
    """
    df_cleaned = df.copy()

    # Preencher 'Age' com a mediana
    median_age = df_cleaned['Age'].median()
    df_cleaned['Age'].fillna(median_age, inplace=True)

    # Preencher 'Embarked' com a moda (valor mais comum) 
    mode_embarked = df_cleaned['Embarked'].mode()[0]
    df_cleaned['Embarked'].fillna(mode_embarked, inplace=True)

    # Preencher qualquer 'Fare' nulo com a mediana
    median_fare = df_cleaned['Fare'].median()
    df_cleaned['Fare'].fillna(median_fare, inplace=True)

    # Converter colunas categóricas para numéricas
    df_cleaned['Sex'] = df_cleaned['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df_cleaned['Embarked'] = df_cleaned['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Remover colunas que não usaremos conforme a descrição da atividade
    # (Name, Ticket, Cabin, PassengerId)
    df_cleaned = df_cleaned.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

    return df_cleaned


def discretize_for_id3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretiza colunas contínuas ('Age', 'Fare') para o uso no ID3.
    Isso converte números em categorias.

    Args:
        df: DataFrame do Titanic já limpo.

    Returns:
        DataFrame com as colunas contínuas discretizadas.
    """
    df_discretized = df.copy()

    # Discretizar 'Age' em faixas etárias(Criança(0-12), Adolescente(12-18), Adulto(18-60), (Idoso(60-100))
    df_discretized['Age'] = pd.cut(
        df_discretized['Age'],
        bins=[0, 12, 18, 60, 100],
        labels=['Criança', 'Adolescente', 'Adulto', 'Idoso']
    )

    # Discretizar 'Fare' usando quantis (divisões com o mesmo número de pessoas)
    # pois a distribuição de 'Fare' é muito assimétrica.
    df_discretized['Fare'] = pd.qcut(
        df_discretized['Fare'],
        q=4,
        labels=['Muito Baixo', 'Baixo', 'Médio', 'Alto']
    )

    return df_discretized