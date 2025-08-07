import requests
from bs4 import BeautifulSoup as bs4
import os
import pandas as pd
import streamlit as st


def importar_ipca_sidra():
    url = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/all/p/all/d/v63%202,v69%202,v2266%2013,v2263%202,v2264%202,v2265%202?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        dados = response.json()
        pd.DataFrame(dados).to_csv('raw_data/ipca_sidra.csv', index=False)
    else:
        st.error(f"Erro ao acessar a API do SIDRA: {response.status_code}")
        return None
    
def importar_tickers():
    st.subheader('Importando tickers')
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')

    # -- ETFs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/
    df_etfs = pd.read_csv('raw_data/etfsListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo
    df_etfs = df_etfs.rename('ticker').to_frame().reset_index(drop=True)
    st.write(f'Qtde. de ETFs: {df_etfs.shape[0]}')
    df_etfs['grupo'] = 'ETF'
    df_etfs['ticker'] = df_etfs['ticker'].apply(lambda x: str(x) + '11')
    
    # -- FIIs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/fundos-de-investimentos/fii/fiis-listados/
    df_fiis = pd.read_csv('raw_data/fiisListados.csv', 
                               encoding='latin-1', 
                               sep=';'
                               ).Fundo
    df_fiis = df_fiis.rename('ticker').to_frame().reset_index(drop=True)
    st.write(f'Qtde. de FIIs:  {df_fiis.shape[0]}')
    df_fiis['grupo'] = 'FII'
    df_fiis['ticker'] = df_fiis['ticker'].apply(lambda x: str(x) + '11')
    
    # -- AÇÕES --
    url = "https://www.dadosdemercado.com.br/acoes"
    try:
        request = requests.get(url)
        content = request.content.decode('utf-8')
        soup = bs4(content, 'html.parser')
        table_container = soup.find('div', {'class': 'table-container'})
        table = table_container.find('table') if table_container else None
        if table is None:
            raise ValueError("Tabela não encontrada no conteúdo HTML.")
        df_acoes = pd.DataFrame(columns=['ticker', 'grupo'])
        lista_acoes = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                ticker = cols[0].text.strip()
                lista_acoes.append(ticker)
        df_acoes = pd.DataFrame(lista_acoes, columns=['ticker'])
        df_acoes['grupo'] = 'Ação'
    except Exception as e:
        st.error(f"Erro ao buscar dados de ações: {e}")
        df_acoes = pd.DataFrame(columns=['ticker', 'grupo'])
    st.write(f'Qtde. de ações:  {df_acoes.shape[0]}')

    # -- ÍNDICES B3 --
    try:
        url = "https://www.dadosdemercado.com.br/b3"
        request = requests.get(url)
        content = request.content.decode('utf-8')
        soup = bs4(content, 'html.parser')
        table_container = soup.find('div', {'class': 'table-container'})
        table = table_container.find('table') if table_container else None
        df_indices = pd.DataFrame(columns=['ticker','grupo'])
        lista_indices = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                ticker = cols[0].text.strip()
                lista_indices.append(ticker)
        df_indices = pd.DataFrame(lista_indices, columns=['ticker'])
        df_indices['grupo'] = 'Índice'
    except Exception as e:
        st.error(f"Erro ao buscar dados de índices: {e}")
        df_indices = pd.DataFrame(columns=['ticker', 'grupo'])
    st.write(f'Qtde. de índices:  {df_indices.shape[0]}')

    # -- Índices macroeconômicos --
    #importar_ipca_sidra()

    # -- API BrAPI - stocks --
    st.write('Importando ações da API BrAPI -- stocks e indexes')
    df_acoes_brapi = pd.DataFrame(columns=['ticker','grupo'])
    df_indices_brapi = pd.DataFrame(columns=['ticker','grupo'])
    url = "https://brapi.dev/api/available"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df_acoes_brapi['ticker'] = data['stocks']
            df_acoes_brapi['grupo'] ='Ações'
            st.write(f"Tickers Ações brapi importados com sucesso: {df_acoes_brapi.shape[0]}")
            df_indices_brapi['ticker'] = data['indexes']
            df_indices_brapi['grupo'] = 'Índice'
            st.write(f"Tickers Índices brapi importados com sucesso: {df_indices_brapi.shape[0]}")
            #Classifica os tickers
            df_acoes_brapi['grupo'] = df_acoes_brapi.apply(lambda x: 'BDR' if x.ticker.endswith('34') or
                                    x.ticker.endswith('39') else 'Ação', axis=1)
            df_acoes_brapi['grupo'] = df_acoes_brapi.apply(lambda x: 'ETF/FII' if x.ticker.endswith('11')
                                else x.grupo, axis=1)
            df_acoes_brapi['grupo'] = df_acoes_brapi.apply(lambda x: 'Índice' if x.ticker.startswith('^') 
                                else x.grupo, axis=1)
        else:
            st.error(f"Erro ao acessar a API BrAPI: {response.status_code}")
    except Exception as e:
        st.error(f'Erro ao buscar dados de BRAPI: {e}')


    # Anexa cada dataframe a df_total se não já tiver
    if 'df_total' not in locals():
        df_total = pd.DataFrame(columns=['ticker', 'grupo'])
    lista_dfs = [df_acoes, df_fiis, df_etfs, df_indices, df_acoes_brapi, df_indices_brapi]
    for df in lista_dfs:
        #adicionar linha se não for repetido
        df_total = pd.concat([df_total, df[~df.ticker.isin(df_total.ticker)]], ignore_index=True)
    
    # adiciona .SA aos tickers se não começar com '^'
    df_total['ticker'] = df_total.apply(lambda x: x.ticker + '.SA' if not x.ticker.startswith('^') else x.ticker, axis=1)


    # -- Adiciona criptomoedas--
    url = f"https://brapi.dev/api/v2/crypto/available"
    df_criptos = pd.DataFrame(columns=['ticker', 'grupo'])
    response = requests.get(url)
    try:
        if response.status_code == 200:
            data = response.json()
            df_criptos['ticker'] = data['coins']
            df_criptos['grupo'] = 'Criptomoeda'
            st.write(f"Criptomoedas importadas da API BrAPI: {df_criptos.shape[0]}")
        else:
            st.error(f"Erro ao acessar a API BrAPI: {response.status_code}")
    except Exception as e:
        st.error(f'Não foi possível importar dados de cripto: {e}')
    df_total = pd.concat([df_total, df_criptos], ignore_index=True)
    
    # -- Moedas --
    # url = https://brapi.dev/api/v2/currency/available

    # -- Inflação --
    # url = https://brapi.dev/api/v2/inflation/available
        
    # -- Taxa de Juros --
    # url = https://brapi.dev/api/v2/prime-rate/available
    
    # -- Mostra quantidade por grupo --
    df_grupos = df_total.groupby('grupo').size().reset_index(name='quantidade')
    st.dataframe(df_grupos)
    st.success(f'Qtde. Total:  {df_total.shape[0]}')

    # -- Exporta os tickers --
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
    df_total.to_csv('raw_data/tickers.csv', index=False, encoding='utf-8')

    

if __name__ == "__main__":
    importar_tickers()
