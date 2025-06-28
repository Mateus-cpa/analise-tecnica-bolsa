import requests
from bs4 import BeautifulSoup as bs4
import pandas as pd
import os


def importar_tickers():
    #criar pasta raw_data se não existir
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')

    tickers = []

    # -- AÇÕES --
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
            tickers.append(ticker)
    
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
    # Extraindo os tickers do conteúdo HTML
    for row in table.find_all('tr')[1:]:  # Ignora o cabeçalho
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers.append(ticker)

    tickers = list(set(tickers))  # Remove duplicatas
    # Salvando os tickers em um arquivo CSV
    with open('raw_data/tickers.csv', 'w', encoding='utf-8') as file:
        for ticker in tickers:
            file.write(f"{ticker}\n")

if __name__ == "__main__":
    importar_tickers()
    print("Tickers definidos com sucesso.")