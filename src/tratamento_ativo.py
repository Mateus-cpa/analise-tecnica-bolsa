import pandas as pd

def detectar_mudanca_tendencia(row, previous_row):
    """Detecta mudanças de tendência com base nas médias móveis.
    Args:
        row (pd.Series): Linha atual do DataFrame.
        previous_row (pd.Series): Linha anterior do DataFrame.
    Returns:
        str: 'Alta' se houve uma mudança de tendência de baixa para alta,
             'Baixa' se houve uma mudança de tendência de alta para baixa,
             None caso contrário.
    """
    mm5_atual = row['MM5']
    mm21_atual = row['MM21']
    mm5_ant = previous_row['MM5']
    mm21_ant = previous_row['MM21']

    if pd.isna(mm5_atual) or pd.isna(mm21_atual):
        return None
    if (mm5_atual > mm21_atual) and (mm5_ant <= mm21_ant):
        return 'Alta'
    elif (mm5_atual < mm21_atual) and (mm5_ant >= mm21_ant):
        return 'Baixa'
    return None

def enriquecer_dados(acao):
    """Enriquece os dados da ação com médias móveis e outros indicadores.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação.
    Returns:
        pd.DataFrame: DataFrame enriquecido com médias móveis e outros indicadores.
    """
        # Corrige o último dado de 'Low' se estiver vazio
    

    if pd.isna(acao.iloc[-1,0]):
        if (acao.iloc[-1,3] < acao.iloc[-1,0]):
            acao.iloc[-1,2] = acao.iloc[-1,0]
        else:
            acao.iloc[-1,2] = acao.iloc[-1,0]

               
    #adiciona média móvel de 5, 72 e 200 períodos
    acao['MM5'] = acao['Close'].rolling(window=5).mean()
    acao['MM21'] = acao['Close'].rolling(window=21).mean()
    acao['MM72'] = acao['Close'].rolling(window=72).mean()
    acao['MM200'] = acao['Close'].rolling(window=200).mean()

    #análise quando MM5 passa por MM72
    acao['mudanca_tendencia'] = None  # Initialize the new column with empty strings

    #calcula RSI
    #acao['rsi'] = RSI(acao['Close'], timeperiod=14)

    
    # Iterar no dataframe e marcar mudanças de tendência
    acao['Date'] = acao.index  # Adiciona a coluna de data
    acao = acao.reset_index(drop=True)
    acao['mudanca_tendencia'] = acao.apply(lambda row: detectar_mudanca_tendencia(row, acao.iloc[row.name - 1]) if row.name > 0 else None, axis=1)

    
    # Adicionar marcadores de topos e fundos em comparação a +/- 5 dias
    acao['marcador'] = None  # Cria a coluna se não existir
    for i in range(5, len(acao) - 5):
        close_atual = acao.iloc[i]['Close']
        prev_5 = acao.iloc[i-5:i]['Close']
        next_5 = acao.iloc[i+1:i+6]['Close']
        # Verifica se o preço atual é maior que os 5 anteriores e os 5 seguintes
        if all(close_atual > x for x in prev_5) and all(close_atual > x for x in next_5):
            acao.at[acao.index[i], 'marcador'] = 'topo'
        elif all(close_atual < x for x in prev_5) and all(close_atual < x for x in next_5):
            acao.at[acao.index[i], 'marcador'] = 'fundo'

    # Convertendo o índice para o formato de data
    acao.index = pd.to_datetime(acao.index)
    acao = acao.set_index('Date')  # Define a coluna de data como índice

    
    return acao

def marcador_hoje(acao):
    # Adiciona marcador no último dia útil real (não previsão)
    acao['marcador_hoje'] = None
    # Encontra o último índice onde Close não é nulo
    ultimo_util = acao[acao['Close'].notnull()].index.max()
    acao.at[ultimo_util, 'marcador_hoje'] = 'hoje'
    return acao


if __name__ == "main":
    detectar_mudanca_tendencia()
    enriquecer_dados()
    marcador_hoje()