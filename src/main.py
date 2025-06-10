#bibliotecas nativas
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Technical Analysis - TA-Lib
#from talib import RSI # type: ignore[import]

# API da Yahoo Finance
import yfinance as yf

import streamlit as st


# machine learning
#from sklearn.feature_selection import SelectKBest
#from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn.neural_network import MLPRegressor
#from sklearn.preprocessing import MinMaxScaler, StandardScaler # para normalizar os dados
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM, Dropout

def configuracoes_iniciais():
    # Configurações iniciais
    plt.style.use('dark_background')  # Corrigido: 'darkgrid' não existe, use 'dark_background'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def definir_ticker():
    """url = "https://www.dadosdemercado.com.br/acoes"
    request = requests.get(url)
    tickers_yf = [ticker + '.SA' for ticker in tickers]
    print(tickers_yf[:10])"""
    ticker = 'BBAS3.SA' 
    #Listando os tickers disponíveis
    return ticker.upper()  # Convertendo para maiúsculas para padronização

def baixar_dados(ticker = None, tempo_anos=1):
    """Baixa os dados do Yahoo Finance para o ticker especificado.
    Args:
        ticker (str): O ticker da ação a ser analisada.
        tempo_anos (int): O número de anos de dados a serem baixados.
    Returns:
        pd.DataFrame: DataFrame contendo os dados de preços da ação.
    """
    # Definindo o período de análise
    today = date.today()
    start_date = today - timedelta(days=tempo_anos*365)  # Últimos 365 dias
    end_date = today  # Até hoje
    intervalo = '1d'  # Intervalo diário
    # Carregando os dados do Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date, interval=intervalo)
    return df

def detectar_mudanca_tendencia(row, df):
    idx = row.name
    # Garante que o índice é inteiro sequencial (após reset_index)
    if idx == 0:
        return None
    mm5_atual = row['MM5']
    mm21_atual = row['MM21']
    mm5_ant = df.loc[idx - 1, 'MM5']
    mm21_ant = df.loc[idx - 1, 'MM21']

    if pd.isna(mm5_atual) or pd.isna(mm21_atual):
        return None
    if (mm5_atual > mm21_atual) and (mm5_ant <= mm21_ant):
        return 'Alta'
    elif (mm5_atual < mm21_atual) and (mm5_ant >= mm21_ant):
        return 'Baixa'
    return None

def enriquecer_dados(acao):
    """Enriquece os dados da ação com médias móveis e outros indicadores.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação.
    Returns:
        pd.DataFrame: DataFrame enriquecido com médias móveis e outros indicadores.
    """
        # Corrige o último dado de 'Low' se estiver vazio
    if pd.isna(acao.iloc[-1,0]):
        if (acao.iloc[-1,3] < acao.iloc[-1,0]):
            acao.iloc[-1,2] = acao.iloc[-1,0]
        else:
            acao.iloc[-1,2] = acao.iloc[-1,0]

               
    #adiciona média móvel de 5, 72 e 200 períodos
    acao['MM5'] = acao['Close'].rolling(window=5).mean()
    acao['MM21'] = acao['Close'].rolling(window=21).mean()
    acao['MM72'] = acao['Close'].rolling(window=72).mean()
    acao['MM200'] = acao['Close'].rolling(window=200).mean()

    #análise quando MM5 passa por MM72
    acao['mudanca_tendencia'] = None  # Initialize the new column with empty strings

    #calcula RSI
    #acao['rsi'] = RSI(acao['Close'], timeperiod=14)

    
    #DATAFRAME
    st.subheader("Dados do ativo:")
    st.dataframe(acao.sort_index(ascending=False))
    
    # Iterar no dataframe e marcar mudanças de tendência
    acao = acao.reset_index(drop=True)
    acao['mudanca_tendencia'] = acao.apply(lambda row: detectar_mudanca_tendencia(row, acao), axis=1)

        # Adicionar marcadores de topos e fundos em comparação a +/- 5 dias
    for i in range(5, len(acao)-5):
        date = acao.index[i]
        previous_date = acao.index[acao.index.get_loc(date) - 1]
        next_dates = acao.index[i+1:i+6]

        if all(acao.at[date,'Close'] > acao.at[previous_date,'Close'] for previous_date in previous_dates) and \
        all(acao.at[date,'Close'] > acao.at[next_date,'Close'] for next_date in next_dates):
            acao.at[date,'marcador'] = 'topo'

        if all(acao.at[date,'Close'] < acao.at[previous_date,'Close'] for previous_date in previous_dates) and \
        all(acao.at[date,'Close'] < acao.at[next_date,'Close'] for next_date in next_dates):
            acao.at[date,'marcador'] = 'fundo'

    #Datas com marcadores
    data_mudanca = acao[acao['mudanca_tendencia'].notnull()].index
    data_topo = acao[acao['marcador'] == 'topo'].index
    data_fundo = acao[acao['marcador'] == 'fundo'].index
    
    return acao

def mostrar_dados(tempo_anos=1):
    configuracoes_iniciais()
    ticker = definir_ticker()
    acao = baixar_dados(ticker, tempo_anos)

    enriquecer_dados(acao)


    # Exibindo os dados
    st.title(f"Análise de {ticker.split('.')[0]}")
    

    # Plotando o gráfico de preços em candlestick no streamlit
    st.subheader(f"Gráfico de Preços - {ticker}")
    # Verificando se o DataFrame está vazio
    if acao.empty:
        st.error("Nenhum dado disponível para o ticker selecionado.")
        return
    # Convertendo o índice para o formato de data
    acao.index = pd.to_datetime(acao.index)
    # Verificando se a coluna 'Close' existe
    if 'Close' not in acao.columns:
        st.error("Coluna 'Close' não encontrada nos dados.")
        return
    # Plotando o gráfico de preços

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Preço de Fechamento'))
    fig.update_layout(title=f"Gráfico de Preços - {ticker}", xaxis_title='Data', yaxis_title='Preço (R$)')

    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    # Exibindo estatísticas descritivas
    st.write("Estatísticas Descritivas:")
    st.dataframe(df.describe())


if __name__ == "__main__":
    configuracoes_iniciais()
    tempo_anos = 1
    mostrar_dados(tempo_anos=1)