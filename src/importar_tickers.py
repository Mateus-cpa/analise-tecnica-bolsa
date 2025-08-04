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

    tickers_dict = {}

    # -- ETFs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/
    print('Importando ETFs')
    lista_etfs = pd.read_csv('raw_data/etfsListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    st.write(f'Qtde. de ETFs:  {len(lista_etfs)}')
    for i in lista_etfs:
        ticker = i + '11'
        tickers_dict[ticker] = 'ETF'

    # -- FIIs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/fundos-de-investimentos/fii/fiis-listados/
    print('Importando FIIs')
    lista_fiis = pd.read_csv('raw_data/fiisListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    st.write(f'Qtde. de FIIs:  {len(lista_fiis)}')
    for i in lista_fiis:
        ticker = i + '11'
        tickers_dict[ticker] = 'FII'

    # -- AÇÕES --
    print('Importando ações')
    url = "https://www.dadosdemercado.com.br/acoes"
    request = requests.get(url)
    content = request.content.decode('utf-8')
    soup = bs4(content, 'html.parser')
    table_container = soup.find('div', {'class': 'table-container'})
    table = table_container.find('table') if table_container else None
    if table is None:
        raise ValueError("Tabela não encontrada no conteúdo HTML.")
    qtd_acoes = 0
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers_dict[ticker] = 'Ação'
            qtd_acoes += 1
    st.write(f'Qtde. de Ações:  {qtd_acoes}')

    # -- ÍNDICES B3 --
    print('Importando índices')
    url = "https://www.dadosdemercado.com.br/b3"
    request = requests.get(url)
    content = request.content.decode('utf-8')
    soup = bs4(content, 'html.parser')
    table_container = soup.find('div', {'class': 'table-container'})
    table = table_container.find('table') if table_container else None
    qtd_indices = 0
    if table is None:
        raise ValueError("Tabela não encontrada no conteúdo HTML.")
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers_dict[ticker] = 'Índice'
            qtd_indices += 1
    st.write(f'Qtde. de Índices:  {qtd_indices}')

    # -- Índices macroeconômicos --
    importar_ipca_sidra()

    # -- API BrAPI --
    print
    url = "https://brapi.dev/api/available"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tickers = data['stocks']
        st.write(f"Tickers brapi importados com sucesso: {len(tickers)}")
        #append data['indexes'] a tickers
        tickers += data['indexes']
    else:
        st.write(f"Erro ao acessar a API BrAPI: {response.status_code}")

    df = pd.DataFrame(tickers, columns=['ticker'])
    #Classifica os tickers
    df['grupo'] = 'Sem classificação'  
    df['grupo'] = df.apply(lambda x: 'BDR' if x.ticker.endswith('34') or
                            x.ticker.endswith('39') else 'Ação', axis=1)
    df['grupo'] = df.apply(lambda x: 'ETF/Unit/FII' if x.ticker.endswith('11') 
                           else x.grupo, axis=1)
    df['grupo'] = df.apply(lambda x: 'Índice' if x.ticker.startswith('^') 
                           else x.grupo, axis=1) 
    df['grupo'] = df.apply(lambda x: 'Ação' if x.ticker.endswith('F')
                           else x.grupo, axis=1)

    # Anexa df a tickers_dict se não existir
    qtde_retirados = 0
    for index, row in df.iterrows():
        ticker = row['ticker']
        grupo = row['grupo']
        if ticker not in tickers_dict:
            tickers_dict[ticker] = grupo
        else:
            qtde_retirados += 1
    st.write(f'Qtde. de tickers retirados de BRAPI: {qtde_retirados}')
    
    #Utiliza pandas para informar quantidade por grupo
    df_tickers = pd.DataFrame(list(tickers_dict.items()), 
                              columns=['ticker', 'grupo'])
    df_grupos = df_tickers.groupby('grupo').size().reset_index(name='quantidade')
    st.write(df_grupos)
    

    # -- Exporta os tickers --
    st.write('Exportando tickers para raw_data/tickers.csv')
    # Cria o diretório se não existir
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
    # Salva os tickers em um arquivo CSV
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        file.write('ticker,grupo\n')
        for ticker, grupo in tickers_dict.items():
            file.write(f"{ticker},{grupo}\n")


    # -- TOTAL --
    st.write(f'Qtde. Total (sem duplicatas):  {len(tickers_dict)}')
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        file.write('ticker,grupo\n')
        for ticker, grupo in tickers_dict.items():
            file.write(f"{ticker},{grupo}\n")

if __name__ == "__main__":
    importar_tickers()
