import pandas as pd  #type: ignore
import yfinance as yf #type: ignore
import json


def importar_fundamentos(ticker):
    # Lógica para importar dados fundamentais do ticker
    dados = yf.Ticker(ticker).info
    dados = pd.DataFrame.from_dict(dados, orient='index').T
    #st.write(dados.columns)
    colunas = ['address1','address2','sector','industry','longBusinessSummary',
                'dividendYield','profitMargins','lastDividendValue','lastDividendDate',
                'previousClose','regularMarketChangePercent','quoteType',
                'recommendationKey','targetHighPrice','targetLowPrice',
                'targetMeanPrice','targetMedianPrice','numberOfAnalystOpinions', 
                'customPriceAlertConfidence',
                'symbol','typeDisp','bookValue','priceToBook','marketCap',
                'shortName','longName','website']
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = None
    # cria arquivos json se não existirem
    arquivos_json = ['traducao_setor.json', 'traducao_industria.json']
    for arquivo in arquivos_json:
        try:
            with open(f'bronze_data/{arquivo}', 'r', encoding='utf-8') as f:
                json.load(f)
        except FileNotFoundError:
            with open(f'bronze_data/{arquivo}', 'w', encoding='utf-8') as f:
                json.dump({}, f)

    # Adiciona a coluna setor_pt usando o dicionário de tradução
    try:
        with open('bronze_data/traducao_setor.json', encoding='utf-8') as f:
            traducao_setor = json.load(f)
        setor_en = dados['sector'].values[0]
        setor_pt = traducao_setor.get(setor_en, setor_en)
        dados['setor_pt'] = setor_pt
    except Exception as e:
        dados['setor_pt'] = dados['sector']
    # Adiciona a coluna industria_pt usando o dicionário de tradução
    try:
        with open('bronze_data/traducao_industria.json', encoding='utf-8') as f:
            traducao_industria = json.load(f)
        industria_en = dados['industry'].values[0]
        industria_pt = traducao_industria.get(industria_en, industria_en)
        dados['industria_pt'] = industria_pt
    except Exception as e:
        dados['industria_pt'] = dados['industry']

    return dados


if __name__ == "__main__":
    ticker = 'BBAS3'
    fundamentos = importar_fundamentos(ticker)
