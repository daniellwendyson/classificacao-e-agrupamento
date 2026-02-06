import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Definir o número de pacientes
num_pacientes = 500

# Gerar dados sintéticos
np.random.seed(42)

# Idade: valores entre 18 e 80 anos
idades = np.random.randint(18, 81, num_pacientes)

# IMC: valores entre 18 e 40 (gordo, normal, baixo peso)
imc = np.random.uniform(18, 40, num_pacientes)

# Pressão Arterial: média de 120/80 mmHg, com uma variação aleatória
pressao_arterial = np.random.randint(100, 180, num_pacientes)  # Sistólica
pressao_arterial_diastolica = np.random.randint(60, 120, num_pacientes)  # Diastólica

# Níveis de glicose: entre 70 e 200 mg/dL
glicose = np.random.uniform(70, 200, num_pacientes)

# Colesterol: entre 150 e 300 mg/dL
colesterol = np.random.uniform(150, 300, num_pacientes)

# Criar o DataFrame
dados = pd.DataFrame({
    'idade': idades,
    'IMC': imc,
    'pressao_arterial': pressao_arterial,
    'pressao_arterial_diastolica': pressao_arterial_diastolica,
    'glicose': glicose,
    'colesterol': colesterol
})

# Exibir as primeiras linhas
print(dados.head())

# Salvar o dataset em um arquivo CSV
dados.to_csv('dados_saude_sinteticos.csv', index=False)


import os

# Definir a variável de ambiente antes de importar o KMeans
os.environ["OMP_NUM_THREADS"] = "2"

# Importar kmeans
from sklearn.cluster import KMeans

# Carregar o dataset
data = pd.read_csv('dados_saude_sinteticos.csv')

# Exibir as primeiras linhas do dataset
print(data.head())

# Verificar o resumo estatístico
print(data.describe())

# Verificar valores ausentes
print(data.isnull().sum())

# Selecionar as colunas relevantes
X = data[['idade', 'IMC', 'pressao_arterial', 'glicose', 'colesterol']]

# Normalizar os dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar os dados no espaço reduzido (2D)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Dados de Pacientes')
plt.show()

# Aplicar K-Means (escolher o número de clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Adicionar os resultados ao dataframe original
data['cluster_kmeans'] = clusters_kmeans

# Visualizar os clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('K-Means Clustering')
plt.show()

# Análise estatística dos clusters
print(data.groupby('cluster_kmeans').describe())

# Comparar grupos em relação às variáveis de saúde
for cluster in range(3):  # Se usamos 3 clusters no K-Means
    print(f"\nCluster {cluster}:")
    print(data[data['cluster_kmeans'] == cluster][['idade', 'IMC', 'pressao_arterial', 'glicose', 'colesterol']].mean())


# Foram identificados três clusters. O primeiro grupo, com pacientes obesos e pressão arterial alta, 
# apresenta alto risco cardiovascular e diabetes. O segundo grupo, composto por pacientes com peso saudável,
# tem risco de pré-diabetes e doenças cardíacas. O terceiro grupo, embora jovem e com peso saudável, 
# sofre de hipertensão grave e apresenta risco de doenças metabólicas. Cada cluster reflete diferentes perfis de 
# risco para doenças cardíacas e diabetes.