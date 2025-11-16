import pandas as pd
from linearmodels import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel import compare
import numpy as np

df_filtered = pd.read_csv(r'data\dados_engajamento_painel_final.csv')

df_filtered.drop_duplicates(subset=['Num', 'Semana'], inplace=True)

# Prepara o índice de painel (Entity, Time)
df_panel = df_filtered.set_index(['Num', 'Semana'])

# Normaliza a variável 'Semana'
max_sem = df_panel.index.get_level_values('Semana').max()
min_sem = df_panel.index.get_level_values('Semana').min()
df_panel['Semana_N'] = (df_panel.index.get_level_values('Semana') - min_sem) / (max_sem - min_sem)

# Define a variável dependente (y) e as variáveis independentes (X)
y = df_panel['Engajamento_Agregado']
X = df_panel[['Semana_N']] 

# Alinha X e y, remove NaNs e adiciona o intercepto
y_clean = y.dropna()
X_clean = X.loc[y_clean.index].assign(Intercept=1)

clusters_values = y_clean.index.get_level_values('Num')
cluster_series = pd.Series(clusters_values, index=y_clean.index)

# --- 1. Pooled OLS (Mínimos Quadrados Agrupados) ---
pooled_res = PooledOLS(y_clean, X_clean).fit(
    cov_type='clustered', 
    clusters=cluster_series 
)

# --- 2. Fixed Effects (Efeitos Fixos) ---
fe_res = PanelOLS(y_clean, X_clean, entity_effects=True).fit(
    cov_type='clustered', 
    clusters=cluster_series
)

# --- 3. Random Effects (Efeitos Aleatórios) ---
re_res = RandomEffects(y_clean, X_clean).fit(
    cov_type='clustered', 
    clusters=cluster_series
)

# --- 4. Comparação ---
print("--- Modelos de Dados em Painel (Engajamento Agregado vs. Tendência de Tempo) ---")
print(compare({'Pooled OLS': pooled_res, 'Fixed Effects': fe_res, 'Random Effects': re_res}, stars=True))

# Salva os resultados
results_panel = compare({'Pooled OLS': pooled_res, 'Fixed Effects': fe_res, 'Random Effects': re_res}, stars=True)

with open(r'data\comparacao_modelos_painel.html', 'w') as f:
    f.write(results_panel.__str__())

print("\nEtapa 3 concluída: Modelos de Dados em Painel treinados e comparados. Resultados salvos em 'comparacao_modelos_painel.html'.")