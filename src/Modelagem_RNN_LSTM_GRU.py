import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- 1. Preparação dos Dados ---
df_filtered = pd.read_csv(r'data\dados_engajamento_painel_final.csv')
df_filtered = df_filtered.dropna(subset=['Engajamento_Agregado'])

df_rnn = df_filtered[['Num', 'Semana', 'Engajamento_Agregado']].copy()

# Parâmetros de Sequência
n_steps = 5
n_features = 1 

# --- Função de Criação de Sequências (Time Series Windowing) ---
def create_sequences(data, n_steps):
    X, y = list(), list()
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Escalonamento e Criação de Sequências por Aluno (Entidade)
scaler = MinMaxScaler()
all_X, all_y = [], []

for num in df_rnn['Num'].unique():
    subset = df_rnn[df_rnn['Num'] == num].sort_values('Semana')
    if len(subset) > n_steps:
        
        scaled_data = scaler.fit_transform(subset['Engajamento_Agregado'].values.reshape(-1, 1))
        
        X_aluno, y_aluno = create_sequences(scaled_data, n_steps)
        
        all_X.append(X_aluno)
        all_y.append(y_aluno)

# Concatena todos os resultados
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
X = X.reshape(X.shape[0], X.shape[1], n_features)

# Divisão Treino/Teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# --- 2. Criação dos Modelos RNN (LSTM e GRU) ---
def build_rnn_model(rnn_type):
    model = Sequential()
    if rnn_type == 'LSTM':
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    elif rnn_type == 'GRU':
        model.add(GRU(50, activation='relu', input_shape=(n_steps, n_features)))
    
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 3. Treinamento e Previsão ---
lstm_model = build_rnn_model('LSTM')
lstm_model.fit(X_train, y_train, epochs=10, verbose=0)
lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)

gru_model = build_rnn_model('GRU')
gru_model.fit(X_train, y_train, epochs=10, verbose=0)
gru_pred_scaled = gru_model.predict(X_test, verbose=0)

# --- 4. Desescalonamento e Avaliação ---
def inverse_transform_predictions(predictions, original_data):
    temp_scaler = MinMaxScaler()
    temp_scaler.fit(original_data['Engajamento_Agregado'].values.reshape(-1, 1))
    return temp_scaler.inverse_transform(predictions)

y_test_inverse = inverse_transform_predictions(y_test, df_rnn)
lstm_pred_inverse = inverse_transform_predictions(lstm_pred_scaled, df_rnn)
gru_pred_inverse = inverse_transform_predictions(gru_pred_scaled, df_rnn)

# Cálculo do RMSE
rmse_lstm = np.sqrt(mean_squared_error(y_test_inverse, lstm_pred_inverse))
rmse_gru = np.sqrt(mean_squared_error(y_test_inverse, gru_pred_inverse))

print("\n--- Resultados dos Modelos RNN (LSTM e GRU) ---")
results_rnn = pd.DataFrame({
    'Modelo': ['LSTM', 'GRU'],
    'RMSE': [rmse_lstm, rmse_gru]
})
print(results_rnn.to_markdown(index=False))

print("\nEtapa 4 (RNN) concluída.")