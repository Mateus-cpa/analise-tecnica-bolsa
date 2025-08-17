import os
import numpy as np
import pandas as pd 
import yfinance as yf
import streamlit as st  # Adicione esta linha

from traducao_base import traduzir_base  # Adicione no topo do arquivo


def atualizar_base_setores():
    """
    Atualiza a base de setores, buscando dados do Yahoo Finance, tratando erros, garantindo colunas e exportando resultados.
    """

    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    os.makedirs(raw_data_dir, exist_ok=True)
    tickers_path = os.path.join(raw_data_dir, 'tickers.csv')
    setores_path = os.path.join(raw_data_dir, 'lista_setores.csv')

    # Garante que o arquivo exista com o cabeçalho correto
    colunas = [
        'ticker', 'grupo', 'nome', 'nome completo', 'setor', 'industria',
        'rendimento', 'variacao_valor', 'recomendação', 'confiança do alerta', 
        'tipo',
        'valor patrimonial por ação', 'pvp', 'valor_mercado',
        'endereco1', 'endereco2', 'ultimo_valor_dividendo', 'ultima_data_dividendo',
        'preco_alvo_maximo', 'preco_alvo_minimo', 'preco_alvo_medio', 'preco_alvo_mediana',
        'numero_opinioes_analistas', 'sumario'
    ]
    if not os.path.exists(setores_path):
        pd.DataFrame(columns=colunas).to_csv(setores_path, index=False)

    df_tickers = pd.read_csv(tickers_path)
    #df_tickers = df_tickers.sample(10).reset_index(drop=True)  # Seleciona 10 tickers aleatórios

    setores = []

    progress_bar = st.progress(0)

    for i, row in df_tickers.iterrows():
        ticker = row['ticker']
        grupo = row['grupo']
        
        try:
            info = yf.Ticker(ticker).info # Assumindo que yf está importado
        except Exception as e:
            info = {}
            st.warning(f"Erro ao buscar dados de {ticker}: {e}")
        
        
        # Monta o dicionário do setor diretamente da variável info
        setor_data = {
            'ticker': ticker,
            'grupo': grupo,
            'nome': info.get('shortName'),
            'nome completo': info.get('longName'),
            'setor': info.get('sector'),
            'industria': info.get('industry'),
            'rendimento': info.get('dividendYield'),
            'variacao_valor': info.get('regularMarketChangePercent'),
            'recomendação': info.get('recommendationKey'),
            'confiança do alerta': info.get('customPriceAlertConfidence'),
            'tipo': info.get('typeDisp'),
            'valor patrimonial por ação': info.get('bookValue'),
            'pvp': info.get('priceToBook'),
            'valor_mercado': info.get('marketCap'),
            'endereco1': info.get('address1'),
            'endereco2': info.get('address2'),
            'ultimo_valor_dividendo': info.get('lastDividendValue'),
            'ultima_data_dividendo': info.get('lastDividendDate'),
            'preco_alvo_maximo': info.get('targetHighPrice'),
            'preco_alvo_minimo': info.get('targetLowPrice'),
            'preco_alvo_medio': info.get('targetMeanPrice'),
            'preco_alvo_mediana': info.get('targetMedianPrice'),
            'numero_opinioes_analistas': info.get('numberOfAnalystOpinions'),
            'sumario': info.get('longBusinessSummary')
        }
        setores.append(setor_data)
        st.success(f'{ticker} - {setor_data["nome"]} ({grupo}) - {setor_data["setor"]} - {setor_data["industria"]} - {setor_data["tipo"]} - DY: {setor_data["rendimento"]}% - Variação: {setor_data["variacao_valor"]}%')
        progress_bar.progress((i + 1) / len(df_tickers))

        df_setores = pd.DataFrame(setores)

        # Garante que todas as colunas esperadas existem
        for i, row in df_tickers.iterrows():
            ticker = row['ticker']
            grupo = row['grupo']
            try:
                info = yf.Ticker(ticker).info
            except Exception as e:
                st.warning(f"Erro ao buscar dados de {ticker}: {e}")
                info = {}
            setor_data = {
                'ticker': ticker,
                'grupo': grupo,
                'nome': info.get('shortName'),
                'nome completo': info.get('longName'),
                'setor': info.get('sector'),
                'industria': info.get('industry'),
                'rendimento': info.get('dividendYield'),
                'variacao_valor': info.get('regularMarketChangePercent'),
                'recomendação': info.get('recommendationKey'),
                'confiança do alerta': info.get('customPriceAlertConfidence'),
                'tipo': info.get('typeDisp'),
                'valor patrimonial por ação': info.get('bookValue'),
                'pvp': info.get('priceToBook'),
                'valor_mercado': info.get('marketCap'),
                'endereco1': info.get('address1'),
                'endereco2': info.get('address2'),
                'ultimo_valor_dividendo': info.get('lastDividendValue'),
                'ultima_data_dividendo': info.get('lastDividendDate'),
                'preco_alvo_maximo': info.get('targetHighPrice'),
                'preco_alvo_minimo': info.get('targetLowPrice'),
                'preco_alvo_medio': info.get('targetMeanPrice'),
                'preco_alvo_mediana': info.get('targetMedianPrice'),
                'numero_opinioes_analistas': info.get('numberOfAnalystOpinions'),
                'sumario': info.get('longBusinessSummary')
            }
            setores.append(setor_data)
            st.success(f'{ticker} - {setor_data["nome"]} ({grupo}) - {setor_data["setor"]} - {setor_data["industria"]} - {setor_data["tipo"]} - DY: {setor_data["rendimento"]}% - Variação: {setor_data["variacao_valor"]}%')
            progress_bar.progress((i + 1) / len(df_tickers))

        # Após o loop, cria o DataFrame e faz todo o tratamento
        df_setores = pd.DataFrame(setores)

        # Garante que todas as colunas esperadas existem
        colunas = [
            'ticker', 'grupo', 'nome', 'nome completo', 'setor', 'industria',
            'rendimento', 'variacao_valor', 'recomendação', 'confiança do alerta', 'tipo',
            'valor patrimonial por ação', 'pvp', 'valor_mercado',
            'endereco1', 'endereco2', 'ultimo_valor_dividendo', 'ultima_data_dividendo',
            'preco_alvo_maximo', 'preco_alvo_minimo', 'preco_alvo_medio', 'preco_alvo_mediana',
            'numero_opinioes_analistas', 'sumario'
        ]
        for col in colunas:
            if col not in df_setores.columns:
                df_setores[col] = None

        # Salva em CSV em primeira instância
        df_setores.to_csv(setores_path, index=False)

        # Calcula coluna 'expectativa' = (preco_alvo_medio - valor_mercado) / valor_mercado * 100
        df_setores['expectativa'] = (pd.to_numeric(df_setores['preco_alvo_medio'], errors='coerce') - pd.to_numeric(df_setores['valor_mercado'], errors='coerce')) / pd.to_numeric(df_setores['valor_mercado'], errors='coerce') * 100

        # Retira dados com nome NA
        df_setores = df_setores.dropna(subset=['nome'])

        # Remove prefixos indesejados da coluna industria
        for prefix in ['Utilities - ', 'Real Estate - ', 'REIT - ']:
            df_setores['industria'] = df_setores['industria'].str.replace(prefix, '', regex=False)

        # Preenche valores nulos de 'rendimento' com 0.0
        df_setores['rendimento'] = pd.to_numeric(df_setores['rendimento'], errors='coerce').fillna(0.0)

        # Trata valores ETF/FIAGRO
        df_setores['grupo'] = df_setores.apply(lambda x: 'Fiagro' if 'Fiagro' in str(x['nome completo']) else x['grupo'], axis=1)

        # Exportação do DataFrame para CSV após tratamento
        df_setores.to_csv(setores_path, index=False)
        st.success("Importação realizada com sucesso.")

        traduzir_base()
        st.success("Base de dados traduzida com sucesso.")


