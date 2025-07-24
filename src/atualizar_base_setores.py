import os
import numpy as np
import pandas as pd 
import yfinance as yf
import streamlit as st  # Adicione esta linha

from traducao_base import traduzir_base  # Adicione no topo do arquivo


def atualizar_base_setores():
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    os.makedirs(raw_data_dir, exist_ok=True)
    tickers_path = os.path.join(raw_data_dir, 'tickers.csv')
    setores_path = os.path.join(raw_data_dir, 'lista_setores.csv')

    # Garante que o arquivo exista com o cabeçalho correto
    if not os.path.exists(setores_path):
        colunas = [
            'ticker', 'grupo', 'nome', 'nome completo', 'setor', 'industria',
            'rendimento', 'variacao_valor', 'recomendação', 'confiança do alerta', 'tipo'
        ]
        pd.DataFrame(columns=colunas).to_csv(setores_path, index=False)

    df_tickers = pd.read_csv(tickers_path)
    #df_tickers = df_tickers.sample(50).reset_index(drop=True)  # Seleciona 50 tickers aleatórios
    setores = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in df_tickers.iterrows():
        ticker = row['ticker']
        grupo = row['grupo']
        try:
            info = yf.Ticker(ticker + '.SA').info # Assumindo que yf está importado
            nome = info.get('shortName') or ''
            nome_completo = info.get('longName') or ''
            setor = info.get('sector') or ''
            industria = info.get('industry') or ''
            rendimento = info.get('dividendYield') or ''
            variacao_valor = info.get('regularMarketChangePercent') or ''
            recomendacao = info.get('recommendationKey') or ''
            confianca_alerta = info.get('customPriceAlertConfidence') or ''
            tipo = info.get('typeDisp') or ''
            status_text.text(f"Ticker: {ticker} | Grupo: {grupo} | Nome: {nome} | Setor: {setor} | Indústria: {industria}")
        except Exception as e:
            print(f"Erro ao buscar dados de {ticker}: {e}")
            nome = ''
            nome_completo = ''
            setor = ''
            industria = ''
            rendimento = ''
            variacao_valor = ''
            recomendacao = ''
            confianca_alerta = ''
            tipo = ''

        setores.append({'ticker': ticker, 
                        'grupo': grupo,
                        'nome': nome,
                        'nome completo' : nome_completo,
                        'setor': setor, 
                        'industria': industria,
                        'rendimento': rendimento,
                        'variacao_valor' : variacao_valor,
                        'recomendação': recomendacao,
                        'confiança do alerta': confianca_alerta,
                        'tipo': tipo})
        progress_bar.progress((i + 1) / len(df_tickers))

    df_setores = pd.DataFrame(setores)
    
    #retirar dados com nome NA
    df_setores = df_setores.dropna(subset=['nome'])

    #retirar 'REIT -', 'Utilities -' e 'Real Estate - ' de setores_df['industria']
    df_setores['industria'] = df_setores['industria'].str.replace('Utilities - ', '', regex=False)
    df_setores['industria'] = df_setores['industria'].str.replace('Real Estate - ', '', regex=False)
    df_setores['industria'] = df_setores['industria'].str.replace('REIT - ', '', regex=False)
    # Preenchendo valores nulos de 'rendimento' com numpy
    df_setores['rendimento'] = np.where(df_setores['rendimento'].isnull(), 0.0, df_setores['rendimento'])

    df_setores.to_csv(setores_path, index=False)
    st.success("Importação realizada com sucesso.")
    
    traduzir_base()
    st.success("Base de dados traduzida com sucesso.")


