import os
import pandas as pd 
import yfinance as yf
import streamlit as st  # Adicione esta linha

def atualizar_base_setores():
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    os.makedirs(raw_data_dir, exist_ok=True)
    tickers_path = os.path.join(raw_data_dir, 'tickers.csv')
    setores_path = os.path.join(raw_data_dir, 'lista_setores.csv')

    df_tickers = pd.read_csv(tickers_path)
    setores = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in df_tickers.iterrows():
        ticker = row['ticker']
        grupo = row['grupo']
        try:
            info = yf.Ticker(ticker + '.SA').info
            nome = info.get('shortName','N/A')
            nome_completo = info.get('longName','N/A')
            setor = info.get('sector', 'N/A')
            industria = info.get('industry', 'N/A')
            rendimento = info.get('dividendYield','N/A')
            variacao_valor = info.get('regularMarketChangePercent','N/A')
            recomendacao = info.get('recommendationKey','N/A')
            confianca_alerta = info.get('customPriceAlertConfidence','N/A')
            tipo = info.get('typeDisp','N/A')
            status_text.text(f"Ticker: {ticker} | Grupo: {grupo} | Nome: {nome} | Setor: {setor} | Indústria: {industria}")
        except Exception:
            nome = 'N/A'
            nome_completo = 'N/A'
            setor = 'N/A'
            industria = 'N/A'
            rendimento = 'N/A'
            variacao_valor = 'N/A'
            recomendacao = 'N/A'
            confianca_alerta = 'N/A'
            tipo = 'N/A'

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
    df_setores.to_csv(setores_path, index=False)
    st.success("Importação realizada com sucesso!")

