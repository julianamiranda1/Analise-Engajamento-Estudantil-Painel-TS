import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- 1. Preparação dos Dados ---
engagement_ts_filtered = pd.read_csv(r'data\engajamento_medio_semanal.csv')
# Filtra para as 13 semanas consistentes e define 'Semana' como índice
engagement_ts_filtered = engagement_ts_filtered[engagement_ts_filtered['Semana'] <= 13].set_index('Semana')

ts_data = engagement_ts_filtered['Engajamento_Agregado']

# Divisão em Treinamento e Teste (usando as últimas 3 semanas para teste)
forecast_periods = 3
train = ts_data.iloc[:-forecast_periods]
test = ts_data.iloc[-forecast_periods:] 
test_index = test.index

# --- 2. Modelo Holt-Winters (Suavização Exponencial) ---
try:
    hw_model = ExponentialSmoothing(
        train, 
        trend='add', 
        seasonal=None, 
        initialization_method="estimated"
    ).fit()
    hw_forecast = hw_model.forecast(forecast_periods)
except Exception as e:
    # Se houver um erro, tenta um modelo mais simples (fallback)
    print(f"Erro no Holt-Winters: {e}")
    hw_forecast = pd.Series([train.iloc[-1]] * forecast_periods, index=test_index)


# --- 3. Modelo ARIMA (AutoRegressive Integrated Moving Average) ---
try:
    arima_model = ARIMA(train, order=(1, 1, 0)).fit()
    arima_forecast = arima_model.forecast(forecast_periods)
except Exception as e:
    # Se o ARIMA falhar, usa um modelo mais simples
    print(f"Erro no ARIMA: {e}")
    try:
        arima_model = ARIMA(train, order=(1, 0, 0)).fit()
        arima_forecast = arima_model.forecast(forecast_periods)
    except:
        arima_forecast = pd.Series([train.iloc[-1]] * forecast_periods, index=test_index)


# --- 4. Avaliação (RMSE) ---
mse_hw = mean_squared_error(test.values, hw_forecast.values)
rmse_hw = np.sqrt(mse_hw)

mse_arima = mean_squared_error(test.values, arima_forecast.values)
rmse_arima = np.sqrt(mse_arima)

results = pd.DataFrame({
    'Modelo': ['Holt-Winters', 'ARIMA'],
    'RMSE': [rmse_hw, rmse_arima]
})

print("--- Resultados da Previsão (Modelos Clássicos) ---")
print(results.to_markdown(index=False))

# --- 5. Plotagem (Opcional, requer a biblioteca matplotlib) ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values, label='Treinamento', marker='o')
plt.plot(test.index, test.values, label='Real (Teste)', marker='o', color='red')
plt.plot(test_index, hw_forecast.values, label=f'Previsão Holt-Winters (RMSE: {rmse_hw:.4f})', linestyle='--', marker='x')
plt.plot(test_index, arima_forecast.values, label=f'Previsão ARIMA(1,1,0) (RMSE: {rmse_arima:.4f})', linestyle='--', marker='x')

plt.title('Previsão de Engajamento Agregado (Modelos Clássicos)')
plt.xlabel('Semana')
plt.ylabel('Engajamento Médio Agregado (0 a 1)')
plt.legend()
plt.grid(True)
plt.savefig(r'data\previsao_series_classicas.png')
plt.show()

print("\nModelos Clássicos executados. Gráfico salvo como 'previsao_series_classicas.png'.")