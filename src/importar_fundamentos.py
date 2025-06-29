import os
import pandas as pd
import yfinance as yf



def importar_fundamentos(ticker):
    # Lógica para importar dados fundamentais do ticker
    dados = yf.Ticker(ticker).info
    dados = pd.DataFrame.from_dict(dados, orient='index').T
    #st.write(dados.columns)
    colunas = ['address1','address2','sector','industry','longBusinessSummary',
                'dividendYield','profitMargins','lastDividendValue','lastDividendDate',
                'previousClose','quoteType',
                'recommendationKey','targetHighPrice','targetLowPrice',
                'targetMeanPrice','targetMedianPrice','numberOfAnalystOpinions', 
                'customPriceAlertConfidence',
                'shortName','longName','website']
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = None
    return dados

def importar_lista_setores():
    # criar arquivo ../raw_data/lista_setores.csv
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    os.makedirs(raw_data_dir, exist_ok=True)
    tickers_path = os.path.join(raw_data_dir, 'tickers.csv')
    setores_path = os.path.join(raw_data_dir, 'lista_setores.csv')

    # Lê a lista de tickers
    lista_tickers = pd.read_csv(tickers_path, header=None)[0].tolist()
    setores = []

    for ticker in lista_tickers:
        try:
            info = yf.Ticker(ticker + '.SA').info
            setor = info.get('sector', 'N/A')
            industria = info.get('industry', 'N/A')
            rendimento = info.get('dividendYield','N/A')
            recomendacao = info.get('recommendationKey','N/A')
            confianca_alerta = info.get('customPriceAlertConfidence','N/A')
            tipo = info.get('typeDisp','N/A')
            print(f"Ticker: {ticker} | Setor: {setor} | Indústria: {industria}")
        except Exception:
            setor = 'N/A'
            industria = 'N/A'
        setores.append({'ticker': ticker, 
                        'setor': setor, 
                        'industria': industria,
                        'rendimento': rendimento,
                        'recomendação': recomendacao,
                        'confiança do alerta': confianca_alerta,
                        'tipo': tipo})

    # Salva no CSV
    df_setores = pd.DataFrame(setores)
    df_setores.to_csv(setores_path, index=False)

if __name__ == "__main__":
    
    #fundamentos = importar_fundamentos(ticker)
    importar_lista_setores()
    print(f"Importação realizada com sucesso!")
