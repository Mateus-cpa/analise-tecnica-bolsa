import requests
from bs4 import BeautifulSoup as bs4
import os
import pandas as pd


def importar_tickers():
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
    # Extraindo os tickers do conteúdo HTML
    for row in table.find_all('tr')[1:]:  # Ignora o cabeçalho
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            todos_tickers.append(ticker)
    print('Importando índices')

    # -- ÍNDICES B3 --
    url = "https://www.dadosdemercado.com.br/b3"
    request = requests.get(url)
    content = request.content.decode('utf-8')
    #ler html, pegando a table para achar os códigos da b3 com bs4:
    soup = bs4(content, 'html.parser')
    table_container = soup.find('div', {'class': 'table-container'})
    table = table_container.find('table') if table_container else None
    if table is None:
        raise ValueError("Tabela não encontrada no conteúdo HTML.")
    for row in table.find_all('tr')[1:]:  # Ignora o cabeçalho
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip() # Extraindo os tickers do conteúdo HTML
            todos_tickers.append(ticker)

    # -- Índices macroeconômicos --
    # https://www.dadosdemercado.com.br/indices

    # -- FIIs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/fundos-de-investimentos/fii/fiis-listados/
    #20 tabs
    print('Importando FIIs')
    lista_fiis = pd.read_csv('raw_data/fiisListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    for i in lista_fiis:
        todos_tickers.append(i + '11')

    # -- ETFs --
    #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/
    print('Importando ETFs')
    lista_etfs = pd.read_csv('raw_data/etfsListados.csv', 
                               encoding='latin-1', 
                               sep=';').Fundo.tolist()
    for i in lista_etfs:
        todos_tickers.append(i + '11')


    todos_tickers = list(set(todos_tickers))  # Remove duplicatas
    # Salvando os tickers em um arquivo CSV
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        for ticker in todos_tickers:
            file.write(f"{ticker}\n")

if __name__ == "__main__":
    importar_tickers()
    print("Tickers definidos com sucesso.")