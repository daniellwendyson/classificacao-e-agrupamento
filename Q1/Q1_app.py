from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Carregando o dataset Íris
dados = load_iris()
caracteristicas = dados.data  # Características (comprimento/largura das pétalas e sépalas)
classes = dados.target  # Classes das flores

# Dividir os dados em conjunto de treino e teste
caracteristicas_treino, caracteristicas_teste, classes_treino, classes_teste = train_test_split(caracteristicas, classes, test_size=0.2, random_state=42)

# Criar e treinar o modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(caracteristicas_treino, classes_treino)

# Fazer previsões
previsoes = modelo.predict(caracteristicas_teste)

# Avaliar o modelo
acuracia = accuracy_score(classes_teste, previsoes)
print(f'Acurácia: {acuracia:.2f}')
print('Relatório de Classificação:')
print(classification_report(classes_teste, previsoes, target_names=dados.target_names))

# A acurácia deu 100%, o que significa que houve overfitting (os dados foram decorados, e não aprendidos). Uma possível solução seria 
# aumentar a quantidade de dados para treino.