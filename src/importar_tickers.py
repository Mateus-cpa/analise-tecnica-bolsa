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
    #criar pasta raw_data se não existir
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')

    todos_tickers = []

    # -- AÇÕES --
    print('Importando ações')
    url = "https://www.dadosdemercado.com.br/acoes"
    request = requests.get(url)
    content = request.content.decode('utf-8')
    #ler html, pegando a table para achar os códigos da b3 com bs4:
    soup = bs4(content, 'html.parser')
    table_container = soup.find('div', {'class': 'table-container'})
    table = table_container.find('table') if table_container else None
    if table is None:
        raise ValueError("Tabela não encontrada no conteúdo HTML.")
    qtd_acoes = 0
    # Extraindo os tickers do conteúdo HTML
    for row in table.find_all('tr')[1:]:  # Ignora o cabeçalho
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            todos_tickers.append(ticker)
        qtd_acoes += 1
    st.write(f'Qtde. de Ações:  {qtd_acoes}')

    
    # -- ÍNDICES B3 --
    print('Importando índices')
    url = "https://www.dadosdemercado.com.br/b3"
    request = requests.get(url)
    content = request.content.decode('utf-8')
    #ler html, pegando a table para achar os códigos da b3 com bs4:
    soup = bs4(content, 'html.parser')
    table_container = soup.find('div', {'class': 'table-container'})
    table = table_container.find('table') if table_container else None
    qtd_indices = 0
    if table is None:
        raise ValueError("Tabela não encontrada no conteúdo HTML.")
    for row in table.find_all('tr')[1:]:  # Ignora o cabeçalho
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip() # Extraindo os tickers do conteúdo HTML
            todos_tickers.append(ticker)
            qtd_indices += 1
    st.write(f'Qtde. de Índices:  {qtd_indices}')

    # -- Índices macroeconômicos --
    # https://www.dadosdemercado.com.br/indices
    # IPCA
    importar_ipca_sidra()

    # -- FIIs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/fundos-de-investimentos/fii/fiis-listados/
    #20 tabs
    print('Importando FIIs')
    lista_fiis = pd.read_csv('raw_data/fiisListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    st.write(f'Qtde. de FIIs:  {len(lista_fiis)}')
    for i in lista_fiis:
        todos_tickers.append(i + '11')

    # -- ETFs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/
    print('Importando ETFs')
    lista_etfs = pd.read_csv('raw_data/etfsListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    st.write(f'Qtde. de ETFs:  {len(lista_etfs)}')
    for i in lista_etfs:
        todos_tickers.append(i + '11')

    # -- TOTAL --
    todos_tickers = list(set(todos_tickers))  # Remove duplicatas
    st.write(f'Qtde. Total (sem duplicatas):  {len(todos_tickers)}')    
    # Salvando os tickers em um arquivo CSV
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        for ticker in todos_tickers:
            file.write(f"{ticker}\n")

if __name__ == "__main__":
    importar_tickers()
    print("Tickers definidos com sucesso.")