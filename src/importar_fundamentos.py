import os
import pandas as pd
import yfinance as yf
import streamlit as st

def importar_fundamentos(ticker):
    # Lógica para importar dados fundamentais do ticker
    dados = yf.Ticker(ticker).info
    dados = pd.DataFrame.from_dict(dados, orient='index').T
    #st.write(dados.columns)
    colunas = ['address1','address2','sector','longBusinessSummary',
                'dividendYield','profitMargins','lastDividendValue','lastDividendDate',
                'recommendationKey','targetHighPrice','targetLowPrice',
                'targetMeanPrice','targetMedianPrice','numberOfAnalystOpinions', 
                'customPriceAlertConfidence',
                'shortName','longName']
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = None
    """# Salvar setores de cada ticker em raw_data/setores.csv
    os.makedirs('raw_data', exist_ok=True)  # Garante que a pasta existe
    lista_tickers = pd.read_csv('raw_data/tickers.csv', header=None)[0].tolist()
    setores_path = 'raw_data/setores.csv'
    with open(setores_path, 'w') as f:
        for ticker in lista_tickers:
            f.write(yf.Ticker(ticker).info['sector'] + '\n')"""

    return dados

if __name__ == "__main__":
    ticker = "BPAC11.SA"  # Exemplo de ticker
    fundamentos = importar_fundamentos(ticker)
    print(f"Fundamentos para {ticker}:")

    for key, value in fundamentos.items():
        print(f"{key}: {value}")