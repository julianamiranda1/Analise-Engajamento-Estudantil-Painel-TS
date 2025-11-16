import pandas as pd
import numpy as np

df_long = pd.read_csv(r'data\dados_engajamento_longo.csv', sep=',')

# ===============================================
# 1. CODIFICAÇÃO DAS VARIÁVEIS DE ENGAJAMENTO
# ===============================================

# Pre-Class (Binária: Sim=1, Não=0)
df_long['Pre-Class_N'] = df_long['Pre-Class'].replace({'√': 1, 'N': 0})

# P (Presença: Presente=1, Meia Falta=0.5, Ausente=0)
df_long['P_N'] = df_long['P'].replace({'P': 1, '1/2': 0.5, 'A': 0})

# Hw (Lição de Casa: Fez=2, Parcial=1, Não Fez=0)
df_long['Hw_N'] = df_long['Hw'].replace({'√': 2, '+/-': 1, 'N': 0})

# CP (Participação: Excelente=2, Regular=1, Ruim=0)
df_long['CP_N'] = df_long['CP'].replace({':-D': 2, ':-/': 1, ':-&': 0})

# Bh (Comportamento: OK=1, Ruim=0)
df_long['Bh_N'] = df_long['Bh'].replace({':-||': 1, ':-(': 0})

# Garante que todas as novas colunas sejam numéricas
cols_to_convert = ['Pre-Class_N', 'P_N', 'Hw_N', 'CP_N', 'Bh_N']
for col in cols_to_convert:
    df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

# ===============================================
# 2. NORMALIZAÇÃO (0 a 1) E CÁLCULO DO ENG. AGREGADO
# ===============================================

# Normaliza Hw e CP
df_long['Hw_Norm'] = df_long['Hw_N'] / 2.0
df_long['CP_Norm'] = df_long['CP_N'] / 2.0
# As demais (P_N, Pre-Class_N, Bh_N)
df_long['Pre-Class_Norm'] = df_long['Pre-Class_N']
df_long['P_Norm'] = df_long['P_N']
df_long['Bh_Norm'] = df_long['Bh_N']

# Colunas normalizadas a serem usadas para o cálculo do Engajamento Agregado
normalized_metrics = ['Pre-Class_Norm', 'P_Norm', 'Hw_Norm', 'CP_Norm', 'Bh_Norm']

# Calcula o Engajamento Agregado (média das métricas normalizadas)
df_long['Engajamento_Agregado'] = df_long[normalized_metrics].mean(axis=1)

# ===============================================
# 3. FILTRAGEM E SALVAMENTO DOS DADOS LIMPOS
# ===============================================

df_filtered = df_long[df_long['Semana'] <= 13].copy().dropna(subset=['Engajamento_Agregado'])

df_filtered.to_csv(r'data\dados_engajamento_painel_final.csv', index=False)

engagement_ts = df_filtered.groupby('Semana')['Engajamento_Agregado'].mean().reset_index()
engagement_ts.to_csv(r'data\engajamento_medio_semanal.csv', index=False)