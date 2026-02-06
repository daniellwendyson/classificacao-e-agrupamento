import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Definir número de amostras
total_samples = 10000
fraud_ratio = 0.05  # 5% de fraudes
fraud_samples = int(total_samples * fraud_ratio)
normal_samples = total_samples - fraud_samples

# Gerar dados sintéticos com make_classification
X, y = make_classification(n_samples=total_samples, n_features=5, weights=[0.95, 0.05], random_state=42)

# Criar DataFrame
df = pd.DataFrame(X, columns=['valor_transacao', 'latitude', 'longitude', 'historico_usuario', 'horario_compra'])
df['fraude'] = y  # Rótulo de fraude

def ajustar_valores(df):
    df['valor_transacao'] = np.abs(df['valor_transacao']) * 1000  # Converter para valores positivos
    df['latitude'] = np.round(np.random.uniform(-90, 90, total_samples), 6)
    df['longitude'] = np.round(np.random.uniform(-180, 180, total_samples), 6)
    df['historico_usuario'] = np.random.randint(1, 50, total_samples)  # Número de transações anteriores
    df['horario_compra'] = np.random.randint(0, 24, total_samples)  # Horário do dia
    return df

df = ajustar_valores(df)

# Salvar como CSV
df.to_csv('fraudes_cartao.csv', index=False)
print("Base de dados gerada com sucesso!")

# Definir número de amostras
total_samples = 10000
fraud_ratio = 0.05  # 5% de fraudes
fraud_samples = int(total_samples * fraud_ratio)
normal_samples = total_samples - fraud_samples

# Gerar dados sintéticos com make_classification
X, y = make_classification(n_samples=total_samples, n_features=5, weights=[0.95, 0.05], random_state=42)

# Criar DataFrame
df = pd.DataFrame(X, columns=['valor_transacao', 'latitude', 'longitude', 'historico_usuario', 'horario_compra'])
df['fraude'] = y  # Rótulo de fraude

def ajustar_valores(df):
    df['valor_transacao'] = np.abs(df['valor_transacao']) * 1000  # Converter para valores positivos
    df['latitude'] = np.round(np.random.uniform(-90, 90, total_samples), 6)
    df['longitude'] = np.round(np.random.uniform(-180, 180, total_samples), 6)
    df['historico_usuario'] = np.random.randint(1, 50, total_samples)  # Número de transações anteriores
    df['horario_compra'] = np.random.randint(0, 24, total_samples)  # Horário do dia
    return df

df = ajustar_valores(df)


# Pré-processamento
df.fillna(df.mean(), inplace=True)  # Lidar com valores nulos
scaler = StandardScaler()
df[['valor_transacao', 'historico_usuario', 'horario_compra']] = scaler.fit_transform(df[['valor_transacao', 'historico_usuario', 'horario_compra']])

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['fraude']), df['fraude'], test_size=0.2, random_state=42)

# Escolher modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)


# Avaliação
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar como CSV
df.to_csv('fraudes_cartao.csv', index=False)
print("Base de dados gerada e modelo treinado com sucesso!")

# Sim, o modelo teve um bom desempenho para identificar transações legítimas, com uma precisão de 0,95 para a classe "0" 
# (não fraude). Porém, ele teve dificuldades para identificar as fraudes (classe "1"), 
# com precisão de 0,00 e recall de 0,00, o que significa que ele não conseguiu identificar nenhuma fraude corretamente. 
# Isso é comum quando temos um desequilíbrio de classes, ou seja, a quantidade de fraudes no dataset é muito menor do que as transações legítimas,
# o que faz com que o modelo acabe "ignorando" as fraudes, já que ele tende a se concentrar mais na classe majoritária.

# Isso pode ser melhorado com algumas técnicas, como balancear as classes ou ajustar o limiar de decisão, para que o modelo consiga dar mais 
# atenção às fraudes.