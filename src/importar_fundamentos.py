import pandas as pd
import yfinance as yf

def importar_fundamentos(ticker):
    # Lógica para importar dados fundamentais do ticker
    dados = yf.Ticker(ticker).info
    dados = pd.DataFrame.from_dict(dados, orient='index').T
    colunas = ['address1','address2','sector','longBusinessSummary',
                'dividendYield','profitMargins','lastDividendValue','lastDividendDate',
                'recommendationKey','targetHighPrice','targetLowPrice',
                'targetMeanPrice','targetMedianPrice','numberOfAnalystOpinions']
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = None
    return dados

if __name__ == "__main__":
    ticker = "BPAC11.SA"  # Exemplo de ticker
    fundamentos = importar_fundamentos(ticker)
    print(f"Fundamentos para {ticker}:")

    for key, value in fundamentos.items():
        print(f"{key}: {value}")