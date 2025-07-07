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
    print('Importando ETFs')
    lista_etfs = pd.read_csv('raw_data/etfsListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    st.write(f'Qtde. de ETFs:  {len(lista_etfs)}')
    for i in lista_etfs:
        ticker = i + '11'
        tickers_dict[ticker] = 'ETF'

    # -- FIIs --
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

    # -- TOTAL --
    st.write(f'Qtde. Total (sem duplicatas):  {len(tickers_dict)}')
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        file.write('ticker,grupo\n')
        for ticker, grupo in tickers_dict.items():
            file.write(f"{ticker},{grupo}\n")

if __name__ == "__main__":
    importar_tickers()
    print("Tickers definidos com sucesso.")