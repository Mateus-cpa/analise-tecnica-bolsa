import os
import pandas as pd
import yfinance as yf

def atualizar_base_setores():
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

    atualizar_base_setores()
    print(f"Importação realizada com sucesso!")
