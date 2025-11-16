# Análise Preditiva da Dinâmica de Engajamento de Estudantes

Este projeto investiga como o engajamento dos estudantes evolui ao longo das semanas letivas e desenvolve modelos capazes de **prever quedas de participação individual antes que elas ocorram**.

A abordagem combina **modelos econométricos**, **métodos clássicos de séries temporais** e **redes neurais recorrentes (LSTM/GRU)** para oferecer uma visão completa da dinâmica de engajamento.

## 1. Visão Geral do Projeto

Este projeto tem como objetivo central analisar a evolução do engajamento dos estudantes ao longo das semanas letivas e desenvolver modelos capazes de **prever quedas de participação** em nível individual antes que elas se concretizem. Para isso, os dados, que são observações de múltiplos alunos ao longo do tempo (painel), foram analisados através de uma combinação de métodos econométricos estruturais (Painel) e técnicas avançadas de previsão temporal (ARIMA, Holt-Winters, LSTM e GRU).

## 2. Metodologia e Modelos Aplicados

A análise foi dividida em três frentes de modelagem para responder a diferentes questões sobre a dinâmica do engajamento (Engajamento_Agregado, normalizado de 0 a 1).

| Tipo de Análise     | Modelos             | Objetivo                     | Variável                     |
|---------------------|---------------------|------------------------------|------------------------------|
| Estrutural (Painel) | Pooled OLS, FE, RE  | Avaliar se existe uma **tendência sistêmica** (queda ou aumento) no engajamento ao longo do semestre, controlando a heterogeneidade individual. | Engajamento por aluno/semana |
| Previsão Clássic  a | ARIMA, Holt-Winters | Realizar previsões de **curto prazo** para o Engajamento Médio do grupo.     | Média semanal                |
| Previsão Não Linear | LSTM, GRU           | Capturar **padrões complexos e não lineares** no histórico de cada aluno, ideal para emitir alertas de risco de queda de participação individual.          | Sequência individual         |

## 3. Etapas de Processamento

Os dados foram processados em etapas sequenciais para atingir o formato ideal para modelagem de painel e séries temporais:

1. **Transformação Wide-to-Long:** O dataset original (formato largo, com colunas por semana) foi convertido para o formato longo (uma linha por aluno por semana).
2. **Codificação e Limpeza:** Variáveis categóricas (P, √, :-D, etc.) foram codificadas numericamente e normalizadas na escala de 0 a 1.
3. **Cálculo do Engajamento_Agregado:** Foi criada a variável Engajamento_Agregado (média das métricas normalizadas por aluno/semana), usada como variável dependente principal.
4. **Filtragem do Painel:** Os dados foram filtrados para incluir apenas as 13 semanas de aula consistentes, removendo outliers de nota e semanas incompletas.

## 4. Resultados

### I. Análise Estrutural (Modelos de Painel)

A variável explicativa principal é a Semana_N (tendência de tempo normalizada de 0 a 1), que mede a mudança total no engajamento ao longo do semestre.

| Modelo     | Coef. Semana_N | p-value | R² (Within) | Conclusão     |
|------------|----------------|---------|-------------|---------------|
| Pooled OLS | 0.1331         | 0.0219  | 0.0153      | Sugere leve aumento de engajamento no agregado.  |
| FE         | 0.0499         | 0.2549  | 0.0041      | **Não há tendência significativa.** As quedas de engajamento são choques semanais, não um declínio gradual sistêmico. |
| RE         | 0.0693         | 0.1205  | 0.0035      | Resultado similar ao FE, mas menos rigoroso. |

### II. Modelos de Previsão de Séries Temporais (RMSE)

O **RMSE (Root Mean Squared Error)** mede a precisão da previsão, onde valores menores são melhores.

#### A. Previsão de Engajamento Médio Coletivo

Estes modelos são excelentes para prever a média da turma.

| Modelo       |    RMSE   |
|--------------|-----------|
| Holt-Winters | 0.0305791 |
| ARIMA        | 0.030664  |

**Conclusão:** O Holt-Winters é o modelo mais preciso para a previsão de curto prazo do engajamento agregado, com um erro médio de apenas 3 pontos percentuais.

#### B. Previsão de Engajamento Individual (RNNs)

Estes modelos (LSTM e GRU), treinados em janelas de 5 semanas para prever a 6ª, são projetados para prever o comportamento mais ruidoso do aluno individual.

| Modelo | RMSE     |
|--------|----------|
| LSTM   | 0.32417  |
| GRU    | 0.339951 |

**Conclusão:** O modelo LSTM superou o GRU e é a ferramenta mais indicada para a previsão não linear da dinâmica individual, essencial para a emissão de alertas de risco.

## 5. Conclusão Final e Aplicações

O projeto atendeu a todos os objetivos propostos:
1. **Ausência de Declínio Sistêmico:** A análise de Painel (FE) mostrou que não há uma "queda de engajamento" geral e previsível em toda a turma ao longo do semestre.

2. **Foco em Alertas Individuais:** A queda de participação deve ser tratada como um problema de choques e variação individual, e não de tendência.

3. **Melhor Modelo Preditivo:** O modelo **LSTM** demonstrou ser o mais poderoso para essa tarefa, sendo o mais adequado para construir um sistema de alertas que identifique alunos em risco de queda de participação (por exemplo, quando a previsão do LSTM cai abaixo de um limiar de $0.5$ para a próxima semana).

## 6. Estrutura do Repositório

```
├── src/
│   ├── Transformacao_Wide_to_Long.py # Etapa 1
│   ├── Codificacao_e_Limpeza_Engajamento.py # Etapa 2
│   ├── Modelagem_Dados_Painel.py # Etapa 3
│   ├── Modelagem_Series_Classicas.py # Etapa 4 - Parte 1
│   ├── Modelagem_RNN_LSTM_GRU.py # Etapa 4 - Parte 2
│
├── data/
│   ├── Base anonimizada - Eric - PUC-SP.xlsx
│   ├── dados_engajamento_longo.csv
│   ├── engajamento_medio_semanal.csv
│   ├── dados_engajamento_painel_final.csv
│   ├── comparacao_modelos_painel.html
│   ├── previsao_series_classicas.png
│
└── README.md
```