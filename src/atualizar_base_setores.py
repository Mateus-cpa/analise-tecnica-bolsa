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
            'rendimento', 'variacao_valor', 'recomendação', 'confiança do alerta', 'tipo',
            'valor patrimonial por ação', 'pvp', 'valor_mercado',
            'sumario'
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
            info = yf.Ticker(ticker).info # Assumindo que yf está importado
        except Exception as e:
            print(f"Erro ao buscar dados de {ticker}: {e}")
        info = {}
        
        # atributos a importar
        lista_get = ['shortName', 'longName', 'sector', 'industry', 'dividendYield',
                     'regularMarketChangePercent', 'recommendationKey', 'customPriceAlertConfidence',
                    'typeDisp', 'bookValue', 'priceToBook', 'marketCap', 'address1', 'address2',
                    'lastDividendValue', 'lastDividendDate', 'targetHighPrice', 'targetLowPrice',
                    'targetMeanPrice', 'targetMedianPrice', 'numberOfAnalystOpinions','longBusinessSummary']
        for key in lista_get:
            try:
                info[key] = yf.Ticker(ticker).info.get(key, None)
            except Exception as e:
                print(f"Erro ao buscar {key} de {ticker}: {e}")
                
        setores.append({'ticker': ticker,
                        'grupo': grupo,
                        'nome': info['shortName'],
                        'nome completo': info['longName'],
                        'setor': info['sector'],
                        'industria': info['industry'],
                        'rendimento': info['dividendYield'],
                        'variacao_valor': info['regularMarketChangePercent'],
                        'recomendação': info['recommendationKey'],
                        'confiança do alerta': info['customPriceAlertConfidence'],
                        'tipo': info['typeDisp'],
                        'valor patrimonial por ação': info['bookValue'],
                        'pvp': info['priceToBook'],
                        'valor_mercado': info['marketCap'],
                        'endereco1': info['address1'],
                        'endereco2': info['address2'],
                        'ultimo_valor_dividendo': info['lastDividendValue'],
                        'ultima_data_dividendo': info['lastDividendDate'],
                        'preco_alvo_maximo': info['targetHighPrice'],
                        'preco_alvo_minimo': info['targetLowPrice'],
                        'preco_alvo_medio': info['targetMeanPrice'],
                        'preco_alvo_mediana': info['targetMedianPrice'],
                        'numero_opinioes_analistas': info['numberOfAnalystOpinions'],
                        'sumario': info['longBusinessSummary']
                        })
        #mostrar dados apenas do ativo lido, após apagar
        st.success(f'{ticker} - {info["shortName"]} ({grupo}) - {info["sector"]} - {info["industry"]} - {info["typeDisp"]} - DY: {info["dividendYield"]}% - Variação: {info["regularMarketChangePercent"]}%')
        
        # Atualizar a barra de progresso
        progress_bar.progress((i + 1) / len(df_tickers))

    df_setores = pd.DataFrame(setores)

    # calcular coluna 'expectativa' = (preco_alvo_medio - valor_mercado) / valor_mercado * 100
    df_setores['expectativa'] = (df_setores['preco_alvo_medio'] - df_setores['valor_mercado']) / df_setores['valor_mercado'] * 100

    #retirar dados com nome NA
    df_setores = df_setores.dropna(subset=['nome'])

    #retirar 'REIT -', 'Utilities -' e 'Real Estate - ' de setores_df['industria']
    df_setores['industria'] = df_setores['industria'].str.replace('Utilities - ', '', regex=False)
    df_setores['industria'] = df_setores['industria'].str.replace('Real Estate - ', '', regex=False)
    df_setores['industria'] = df_setores['industria'].str.replace('REIT - ', '', regex=False)
    # Preenchendo valores nulos de 'rendimento' com numpy
    df_setores['rendimento'] = np.where(df_setores['rendimento'].isnull(), 0.0, df_setores['rendimento'])

    # Trata valores ETF
    df_setores['grupo'] = df_setores.apply(lambda x: 'Fiagro' if 'Fiagro' in x['nome completo'] else x.grupo, axis=1)
    #df_setores['grupo'] = df_setores.apply(lambda x: 'ETF/FII' if x.ticker.endswith('11') else x.grupo, axis=1)

    # Exportação do DataFrame para CSV
    df_setores.to_csv(setores_path, index=False)
    st.success("Importação realizada com sucesso.")
    
    traduzir_base()
    st.success("Base de dados traduzida com sucesso.")


