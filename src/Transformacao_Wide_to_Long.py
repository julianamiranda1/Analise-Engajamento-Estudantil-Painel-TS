import pandas as pd
import re

file_name = r'data\Base anonimizada - Eric - PUC-SP.xlsx'
# Carrega os dados. O cabeçalho real começa na terceira linha (índice 2).
df = pd.read_excel(file_name, header=2)

# ===============================================
# 1. PREPARAÇÃO E RENOMEAÇÃO DA COLUNA DA AULA 1
# ===============================================

# As métricas de engajamento que variam ao longo do tempo (variáveis stub)
engagement_metrics = ['Pre-Class', 'P', 'Hw', 'CP', 'Bh']

# A primeira aula (Aula 1) não tem sufixo (.0), então renomeamos para consistência.
rename_map = {col: f'{col}.0' for col in engagement_metrics if col in df.columns and '.' not in col}
df.rename(columns=rename_map, inplace=True)

# ===============================================
# 2. DEFINIÇÃO DE VARIÁVEIS ID E STUB
# ===============================================

# Variáveis ID (não variam no tempo e identificam o aluno)
id_vars = ['Num', 'NOME COMPLETO']

# Variáveis 'stub' (variáveis de engajamento que se repetem a cada aula)
stubnames = engagement_metrics

# Variáveis de tempo. O padrão da coluna é 'StubName.Número' (ex: P.1, Hw.10)
time_series_pattern = re.compile(r'(' + '|'.join(re.escape(s) for s in stubnames) + r')\.\d+$')
time_series_cols = [col for col in df.columns if time_series_pattern.match(col)]

# Incluir todas as colunas que não são séries temporais como IDs (constantes no tempo)
other_non_time_series_cols = [col for col in df.columns if col not in id_vars and col not in time_series_cols]
all_id_vars = id_vars + other_non_time_series_cols

# ===============================================
# 3. TRANSFORMAÇÃO WIDE-TO-LONG
# ===============================================

df_long = pd.wide_to_long(
    df,
    stubnames=stubnames,
    i=all_id_vars,
    j='Aula',
    sep='.',
    suffix=r'\d+'
).reset_index()

# ===============================================
# 4. AJUSTE DO ÍNDICE DE TEMPO E SALVAMENTO
# ===============================================

# O índice 'Aula' (j) foi capturado como 0-indexado (0 a 13).
# Ajustamos para ser 1-indexado (1 a 14) e renomeamos para 'Semana'.
df_long['Aula'] = df_long['Aula'] + 1
df_long.rename(columns={'Aula': 'Semana'}, inplace=True)

# Salva o DataFrame no formato longo
df_long.to_csv(r'data\dados_engajamento_longo.csv', index=False)