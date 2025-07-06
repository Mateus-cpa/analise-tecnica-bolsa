from datetime import date, timedelta
import pandas as pd

import yfinance as yf
import streamlit as st

@st.cache_data(show_spinner="Baixando dados...", max_entries=10)
def baixar_dados(ticker=None, tempo_anos=19):
    """Baixa os dados do Yahoo Finance para o ticker especificado.
    Args:
        ticker (str): O ticker da ação a ser analisada.
        tempo_anos (int): O número de anos de dados a serem baixados.
    Returns:
        pd.DataFrame: DataFrame contendo os dados de preços da ação.
    """
    if not ticker or ticker == 'Nenhum':
        return pd.DataFrame()
    start_date = date.today() - timedelta(days=tempo_anos*365)
    end_date = date.today()
    intervalo = '1d'
    df = yf.download(ticker, start=start_date, end=end_date, interval=intervalo)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    return df