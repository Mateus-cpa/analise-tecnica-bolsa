import pandas as pd  #type: ignore
import yfinance as yf #type: ignore



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
                'shortName','longName','website']
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = None
    return dados


if __name__ == "__main__":
    ticker = 'BBAS3'
    fundamentos = importar_fundamentos(ticker)
