#bibliotecas nativas
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# bibliotecas de terceiros
#from talib import RSI # type: ignore[import] # Technical Analysis - TA-Lib
import yfinance as yf # API da Yahoo Finance
import streamlit as st # Streamlit para interface web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup as bs4  # Importando BeautifulSoup para manipulação de HTML
#from sklearn.feature_selection import SelectKBest
#from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn.neural_network import MLPRegressor
#from sklearn.preprocessing import MinMaxScaler, StandardScaler # para normalizar os dados
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM, Dropout

#bibliotecas locais
from importar_tickers import importar_tickers # Importando a função para definir o ticker


def configuracoes_iniciais():
    # Configurações iniciais
    plt.style.use('dark_background')  # Corrigido: 'darkgrid' não existe, use 'dark_background'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def definir_ticker():
    """Define o ticker a ser analisado.
    Returns:
        str: O ticker da ação a ser analisada.
    """
    st.header("Definir Ticker")
    #importar tickers de raw_data/tickers.csv
    tickers = pd.read_csv('raw_data/tickers.csv', header=None)[0].tolist()
    st.session_state.ticker = st.selectbox(
        "Selecione o ticker da ação",
        options=tickers,
        key="ticker_select"
    )
    ticker = st.session_state.ticker + '.SA'
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
    # Corrige MultiIndex nas colunas, se houver
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]  # Mantém só o 1º nível (ex: 'Close')
    return df

def detectar_mudanca_tendencia(row, previous_row):
    """Detecta mudanças de tendência com base nas médias móveis.
    Args:
        row (pd.Series): Linha atual do DataFrame.
        previous_row (pd.Series): Linha anterior do DataFrame.
    Returns:
        str: 'Alta' se houve uma mudança de tendência de baixa para alta,
             'Baixa' se houve uma mudança de tendência de alta para baixa,
             None caso contrário.
    """
    mm5_atual = row['MM5']
    mm21_atual = row['MM21']
    mm5_ant = previous_row['MM5']
    mm21_ant = previous_row['MM21']

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

    
    # Iterar no dataframe e marcar mudanças de tendência
    acao['Date'] = acao.index  # Adiciona a coluna de data
    acao = acao.reset_index(drop=True)
    acao['mudanca_tendencia'] = acao.apply(lambda row: detectar_mudanca_tendencia(row, acao.iloc[row.name - 1]) if row.name > 0 else None, axis=1)

    
    # Adicionar marcadores de topos e fundos em comparação a +/- 5 dias
    acao['marcador'] = None  # Cria a coluna se não existir
    for i in range(5, len(acao) - 5):
        close_atual = acao.iloc[i]['Close']
        prev_5 = acao.iloc[i-5:i]['Close']
        next_5 = acao.iloc[i+1:i+6]['Close']
        # Verifica se o preço atual é maior que os 5 anteriores e os 5 seguintes
        if all(close_atual > x for x in prev_5) and all(close_atual > x for x in next_5):
            acao.at[acao.index[i], 'marcador'] = 'topo'
        elif all(close_atual < x for x in prev_5) and all(close_atual < x for x in next_5):
            acao.at[acao.index[i], 'marcador'] = 'fundo'

    # Convertendo o índice para o formato de data
    acao.index = pd.to_datetime(acao.index)
    acao = acao.set_index('Date')  # Define a coluna de data como índice

    
    #Datas com marcadores
    data_mudanca = acao[acao['mudanca_tendencia'].notnull()].index
    data_topo = acao[acao['marcador'] == 'topo'].index
    data_fundo = acao[acao['marcador'] == 'fundo'].index
    return acao

def plotar_grafico(acao, ticker):
    """Plota o gráfico de preços da ação com médias móveis e marcadores de tendência.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação enriquecidos.
    """
    # Verifica se o DataFrame está vazio
    if acao.empty:
        st.error("Nenhum dado disponível para o ticker selecionado.")
        return

    # Verificando se a coluna 'Close' existe
    if 'Close' not in acao.columns:
        st.error("Coluna 'Close' não encontrada nos dados.")
        return

    #Criando filtro de data sendo padrão o inicial 30 dias antes do último dado e o final o último dado
    data_inicio, data_fim = st.slider(
        "Selecione o período de análise",
        min_value=acao.index.min().date(),
        max_value=acao.index.max().date(),
        value=(acao.index.max().date() - timedelta(days=30), acao.index.max().date()),
        format="DD/MM/YYYY",
        key="data_slider"
    )
    # Filtra o DataFrame conforme o período selecionado
    acao = acao.loc[(acao.index.date >= data_inicio) & (acao.index.date <= data_fim)]


    # Plotando o gráfico de preços
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=acao.index, y=acao['Close'], mode='lines', name='Preço de Fechamento', marker_color='rgba(255,0,0,0.5)'))

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Adicionando médias móveis
    if col1.checkbox('MM5', value=True):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM5'], mode='lines', name='MM5', marker_color='rgba(250,250,250,0.5)'))
    if col2.checkbox('MM21', value=True):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM21'], mode='lines', name='MM21', marker_color='rgba(160,160,160,0.5)'))
    if col3.checkbox('MM72', value=True):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM72'], mode='lines', name='MM72', marker_color='rgba(90,90,90,0.5)'))
    if col4.checkbox('MM200', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM200'], mode='lines', name='MM200', marker_color='rgba(40,40,40,0.5)'))

    #adicionando marcadores de topos e fundos com scatter
    if col5.checkbox('Topos e Fundos', value=True):
        fig.add_trace(go.Scatter(
            x=acao[acao['marcador'] == 'topo'].index,
            y=acao[acao['marcador'] == 'topo']['Close'],
            mode='markers',
            name='Topos',
            marker=dict(color='red', size=10, symbol='triangle-up')
        ))
        
        fig.add_trace(go.Scatter(
            x=acao[acao['marcador'] == 'fundo'].index,
            y=acao[acao['marcador'] == 'fundo']['Close'],
        mode='markers',
        name='Fundos',
        marker=dict(color='green', size=10, symbol='triangle-down')
        ))
    # Adicionando marcadores de tendência
    if col6.checkbox('Mudança de Tendência', value=True):
        # Filtra as datas com mudança de tendência
        mudanca_tendencia = acao[acao['mudanca_tendencia'].notnull()]
        fig.add_trace(go.Scatter(
            x=mudanca_tendencia.index,
            y=mudanca_tendencia['Close'],
            mode='markers',
            name='Mudança de Tendência',
            marker=dict(color='yellow', size=10, symbol='circle')
        ))
        for i in range(len(acao)):
            if acao.iloc[i]['mudanca_tendencia'] == 'Alta':
                fig.add_annotation(x=acao.index[i], 
                                y=acao.iloc[i]['Close'], 
                                text="Alta", 
                                showarrow=True, 
                                arrowhead=2, 
                                ax=-20, 
                                ay=-30,
                                bgcolor='green',
                                font=dict(color='white'))
            elif acao.iloc[i]['mudanca_tendencia'] == 'Baixa':
                fig.add_annotation(x=acao.index[i], 
                                y=acao.iloc[i]['Close'], 
                                text="Baixa", 
                                showarrow=True, 
                                arrowhead=2, 
                                ax=-20, 
                                ay=30,
                                    bgcolor='red',
                                    font=dict(color='white')
                                )

    
    # Atualizando layout do gráfico
    fig.update_layout(title=f"Gráfico de Preços - {ticker.split('.')[0]}", xaxis_title='Data', yaxis_title='Preço (R$)')
    
    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    st.divider()

    

def lancar_dataframe(acao, ticker):
    """Lança o DataFrame enriquecido no Streamlit.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação enriquecidos.
        ticker (str): O ticker da ação a ser analisada.
    """
    # Verifica se o DataFrame está vazio
    if acao.empty:
        st.error("Nenhum dado disponível para o ticker selecionado.")
        return

    # Exibe o DataFrame no Streamlit
    st.dataframe(acao.sort_index(ascending=False))

    # Exibe o DataFrame como CSV para download
    csv = acao.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker.split('.')[0]}_dados_enriquecidos.csv",
        mime='text/csv'
    )

    # Exibindo estatísticas descritivas
    st.write("Estatísticas Descritivas:")
    st.dataframe(acao.describe())

def mostrar_dados(tempo_anos=1):
    configuracoes_iniciais()
    importar_tickers()  # Importa os tickers disponíveis
    ticker = definir_ticker()
    acao = baixar_dados(ticker, tempo_anos)
    acao = enriquecer_dados(acao)
    st.header(f"Dados do ativo - {ticker.split('.')[0]}")
    plotar_grafico(acao, ticker)
    lancar_dataframe(acao, ticker)
    


if __name__ == "__main__":
    configuracoes_iniciais()
    tempo_anos = 1
    mostrar_dados(tempo_anos=1)