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
#from talib import RSI # https://github.com/ta-lib/ta-lib-python

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
    global ticker, start_date, end_date
    ticker = 'BBAS3.SA' 
    #Listando os tickers disponíveis
    # Baixa a lista de ativos da B3 (arquivo CSV público)
    """url = "https://www.dadosdemercado.com.br/acoes"
    request = requests.get(url)
    tickers_yf = [ticker + '.SA' for ticker in tickers]
    print(tickers_yf[:10])"""
    # Definindo o período de análise
    today = date.today()
    start_date = today - timedelta(days=365)  # Últimos 365 dias
    end_date = today  # Até hoje

def baixar_dados(ticker = None, start_date = None, end_date = None):
    # Carregando os dados do Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def mostrar_dados(df):
    configuracoes_iniciais()
    definir_ticker()
    # Exibindo os dados
    st.title(f"Análise de {ticker}")
    st.write("Dados do ativo:")
    st.dataframe(df)

    # Plotando o gráfico de preços
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Preço de Fechamento'))
    fig.update_layout(title=f"Gráfico de Preços - {ticker}", xaxis_title='Data', yaxis_title='Preço (R$)')
    
    st.plotly_chart(fig)
    
    # Exibindo estatísticas descritivas
    st.write("Estatísticas Descritivas:")
    st.dataframe(df.describe())


if __name__ == "__main__":
    
    df = baixar_dados(ticker, start_date, end_date)
    mostrar_dados(df)