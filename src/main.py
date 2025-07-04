#bibliotecas nativas
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')
import json

# bibliotecas de terceiros
#from talib import RSI # Technical Analysis - TA-Lib
import yfinance as yf # API da Yahoo Finance
import streamlit as st # Streamlit para interface web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay



#bibliotecas locais
from importar_tickers import importar_tickers # Importando a função para definir o ticker
from importar_fundamentos import importar_fundamentos # Importando a função para importar fundamentos
from atualizar_base_setores import atualizar_base_setores
from modelo_preditivo import acao_com_preditivo


def configuracoes_iniciais():
    # Configurações iniciais
    plt.style.use('dark_background')  # Corrigido: 'darkgrid' não existe, use 'dark_background'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def definir_ticker():
    """Define o ticker a ser analisado.
    Returns:
        str: O ticker da ação a ser analisada.
    """
    st.header("Definir Ticker")
    
    # Lê lista_setores.csv com cabeçalho
    setores_df = pd.read_csv('raw_data/lista_setores.csv')  # Espera colunas: ticker, setor, industria

    # Primeira linha de filtros
    col1, col2, col3 = st.columns([0.20, 0.35, 0.45])
    tipo = setores_df['tipo'].unique().tolist() # Filtrar por tipo de ticker - Equity (ações), Funds e Index
    tipo_selecionado = col1.selectbox('Tipo', options=['Todos'] + tipo, key='tipo_select')
    if tipo_selecionado != 'Todos':
        setores_filtrados = setores_df[setores_df['setor'] == tipo_selecionado]['setor'].dropna().unique().tolist()
    else:
        setores_filtrados = setores_df['setor'].unique().tolist() # Filtrar por setor
    setor_selecionado = col2.selectbox('Setor', options=['Todos'] + setores_filtrados, key='setor_select')
    if setor_selecionado != 'Todos': # Filtra subsetores conforme setor
        industrias_filtradas = setores_df[setores_df['setor'] == setor_selecionado]['industria'].dropna().unique().tolist()
    else:
        industrias_filtradas = setores_df['industria'].dropna().unique().tolist()
    industria_selecionada = col3.selectbox('Indústria', options=['Todos'] + industrias_filtradas, key='industrias_select')

    
    # Segunda linha de filtros por nome ou ticker
    # Filtragem apenas pelos selects
    setores_filtrados_df = setores_df.copy()
    if tipo_selecionado != 'Todos':
        setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['tipo'] == tipo_selecionado]
    if setor_selecionado != 'Todos':
        setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['setor'] == setor_selecionado]
    if industria_selecionada != 'Todos':
        setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['industria'] == industria_selecionada]

    # Cria coluna de busca concatenada
    setores_filtrados_df['ticker_busca'] = setores_filtrados_df.apply(
        lambda linha: f"{str(linha['ticker'])} {str(linha.get('nome',''))} {str(linha.get('nome completo',''))}", axis=1
    )

    # Cria dicionário para mapear o texto exibido ao ticker real
    opcoes_dict = {row['ticker_busca']: row['ticker'] for _, row in setores_filtrados_df.iterrows()}
    opcoes_lista = ['Nenhum'] + list(opcoes_dict.keys())

    selecionado = st.selectbox(
        "Ticker da ação",
        options=opcoes_lista,
        key="ticker_select"
    )
    st.session_state.ticker = opcoes_dict.get(selecionado, 'Nenhum')


    ticker = st.session_state.ticker + '.SA' if st.session_state.ticker != 'Nenhum' else 'Nenhum'
    return ticker.upper()  # Convertendo para maiúsculas para padronização

def baixar_dados(ticker = None):
    """Baixa os dados do Yahoo Finance para o ticker especificado.
    Args:
        ticker (str): O ticker da ação a ser analisada.
        tempo_anos (int): O número de anos de dados a serem baixados.
    Returns:
        pd.DataFrame: DataFrame contendo os dados de preços da ação.
    """
    # Definindo o período de análise
    tempo_anos = st.select_slider('Qtde. de anos de download',range(1,20,1),19)
    start_date = date.today() - timedelta(days=tempo_anos*365)  # Últimos 365 dias
    end_date = date.today()  # Até hoje
    #intervalo = st.selectbox('Tempo gráfico',   # Lista Intervalos
    #                         ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'],
    #                         placeholder='1d')
    intervalo = '1d'
    # Carregando os dados do Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date, interval=intervalo)
    # Corrige MultiIndex nas colunas, se houver
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]  # Mantém só o 1º nível (ex: 'Close')
    return df

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

def mostrar_fundamentos(fundamentos: pd.DataFrame):
    """Mostra os fundamentos da ação no Streamlit.
    Args:
        ticker (str): O ticker da ação.
        fundamentos (pd.DataFrame): DataFrame contendo os fundamentos da ação.
    """
    st.header(f"{fundamentos['shortName'].values[0]}")
    st.subheader(f"{fundamentos['longName'].values[0]}")

    st.subheader("Fundamentos")
    if fundamentos.empty:
        st.error("Nenhum dado fundamental disponível para o ticker selecionado.")
        return
    # Exibe os fundamentos no Streamlit
    col1, col2, col3 = st.columns(3)
    with col1:
        #kpis
        if fundamentos['previousClose'].values[0] is not None:
            st.metric("Último Fechamento", f"R$ {fundamentos['previousClose'].values[0]:.2f}")
        if fundamentos['dividendYield'].values[0] is not None:
            st.metric("Dividend Yield", f"{fundamentos['dividendYield'].values[0]/100:.2%}")
        if fundamentos['lastDividendValue'].values[0] is not None:
            st.metric("Último Dividendo", f"R$ {fundamentos['lastDividendValue'].values[0]:.2f}")
        try:
            st.metric("Data do Último Dividendo", f"{pd.to_datetime(fundamentos['lastDividendDate'].values[0]).strftime('%d/%m/%Y')}")
        except Exception as e:
            st.error(f"Erro ao exibir a data do último dividendo: {e}")

    with col2:
        if fundamentos['profitMargins'].values[0] is not None:
            st.metric("Margem de Lucro", f"{fundamentos['profitMargins'].values[0]:.2%}")
        if fundamentos['recommendationKey'].values[0] is not None:
            if 'buy' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Comprar", delta_color="normal")
                #st.metric("Recomendação", ":cow: [Comprar]", delta_color="normal")
            elif 'sell' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Vender", delta_color="inverse")
                #st.metric("Recomendação", ":bear: [Vender]", delta_color="inverse")
        if fundamentos['numberOfAnalystOpinions'].values[0] is not None:
            st.metric('Nº de Opiniões de Analistas', fundamentos['numberOfAnalystOpinions'].values[0])
        if fundamentos['targetMedianPrice'].values[0] is not None:
            st.metric('Mediana de Preço alvo', f"R$ {fundamentos['targetMedianPrice'].values[0]:.2f}")

    with col3:
        if fundamentos['targetHighPrice'].values[0] is not None:
            st.metric('Preço alvo máximo', f"R$ {fundamentos['targetHighPrice'].values[0]:.2f}")
        if fundamentos['targetLowPrice'].values[0] is not None:
            st.metric('Preço alvo mínimo', f"R$ {fundamentos['targetLowPrice'].values[0]:.2f}")
        if fundamentos['targetMeanPrice'].values[0] is not None:
            st.metric('Média de Preço alvo', f"R$ {fundamentos['targetMeanPrice'].values[0]:.2f}")

    mostrar_fundamentos = st.checkbox('Mostrar todos fundamentos')
    if mostrar_fundamentos:
        for col in fundamentos.columns:
            st.write(f"{col}: {fundamentos[col].values[0]}")

def marcador_hoje(acao):
    #adicionar marcador de data de hoje
    acao['marcador_hoje'] = None
    hoje = pd.Timestamp(date.today() - timedelta(days=1)) # Garante que o índice está no formato datetime
    acao.index = pd.to_datetime(acao.index) # Marca a linha correspondente à data de hoje (se existir)
    if hoje in acao.index:
        acao.at[hoje, 'marcador_hoje'] = 'hoje' 
    
    return acao

def plotar_grafico(acao, ticker):
    """Plota o gráfico de preços da ação com médias móveis e marcadores de tendência.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação enriquecidos.
    """
    st.header("Gráfico de Preços")
    # Verifica se o DataFrame está vazio
    if acao.empty:
        st.error("Nenhum dado disponível para o ticker selecionado.")
        return

    # Verificando se a coluna 'Close' existe
    if 'Close' not in acao.columns:
        st.error("Coluna 'Close' não encontrada nos dados.")
        return

    data_inicio, data_fim = st.slider(
        "Selecione o período de análise",
        min_value=acao.index.min().date(),
        max_value=acao.index.max().date(),
        value=(acao.index.max().date() - timedelta(days=17), acao.index.max().date() - timedelta(days=1)),
        format="DD/MM/YYYY",
        key="data_slider")
    
    # Filtra o DataFrame conforme o período selecionado
    acao = acao.loc[(acao.index.date >= data_inicio) & (acao.index.date <= data_fim)]

    
    # Plotando o gráfico de preços
    fig = make_subplots(rows=1, cols=1)
    
    # Adicionando elementos no gráfico
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.checkbox('MM5', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM5'], mode='lines', name='MM5', marker_color='rgba(250,250,250,0.5)'))
    if col1.checkbox('Candles', value=False):
        fig.add_trace(
        go.Candlestick(
            x=acao.index,
            open=acao['Open'],
            high=acao['High'],
            low=acao['Low'],
            close=acao['Close'],
            name='Candlestick'))
    if col2.checkbox('MM21', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM21'], mode='lines', name='MM21', marker_color='rgba(160,160,160,0.5)'))
    if col3.checkbox('MM72', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM72'], mode='lines', name='MM72', marker_color='rgba(90,90,90,0.5)'))
    if col4.checkbox('MM200', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM200'], mode='lines', name='MM200', marker_color='rgba(40,40,40,0.5)'))

    #adicionando marcadores de topos e fundos com scatter
    if col5.checkbox('Topos e Fundos', value=False):
        fig.add_trace(go.Scatter(
            x=acao[acao['marcador'] == 'topo'].index,
            y=acao[acao['marcador'] == 'topo']['Close'],
            mode='markers',
            name='Topos',
            marker=dict(color='red', size=10, symbol='triangle-up')
        ))
        
        fig.add_trace(go.Scatter(
            x=acao[acao['marcador'] == 'fundo'].index,
            y=acao[acao['marcador'] == 'fundo']['Close'],
        mode='markers',
        name='Fundos',
        marker=dict(color='green', size=10, symbol='triangle-down')
        ))
    fig.add_trace(go.Scatter(x=acao.index, y=acao['Close'], mode='lines', name='Real', marker_color='rgba(255,0,0,0.5)'))
    
    # Adicionando marcadores de tendência
    if col2.checkbox('Mudança de Tendência', value=False):
        # Filtra as datas com mudança de tendência
        mudanca_tendencia = acao[acao['mudanca_tendencia'].notnull()]
        fig.add_trace(go.Scatter(
            x=mudanca_tendencia.index,
            y=mudanca_tendencia['Close'],
            mode='markers',
            name='Mudança de Tendência',
            marker=dict(color='yellow', size=10, symbol='circle')
        ))
        for i in range(len(acao)):
            if acao.iloc[i]['mudanca_tendencia'] == 'Alta':
                fig.add_annotation(x=acao.index[i], 
                                y=acao.iloc[i]['Close'], 
                                text="Alta", 
                                showarrow=True, 
                                arrowhead=2, 
                                ax=-20, 
                                ay=-30,
                                bgcolor='green',
                                font=dict(color='white'))
            elif acao.iloc[i]['mudanca_tendencia'] == 'Baixa':
                fig.add_annotation(x=acao.index[i], 
                                y=acao.iloc[i]['Close'], 
                                text="Baixa", 
                                showarrow=True, 
                                arrowhead=2, 
                                ax=-20, 
                                ay=30,
                                    bgcolor='red',
                                    font=dict(color='white')
                                )

    #marcadores de Machine Learning
    try:
        with open('bronze_data/coeficientes_modelos.json', 'r') as f:
            coeficientes_modelos = json.load(f) #salva como dict
    except FileNotFoundError:
        st.warning("Arquivo de coeficientes não encontrado.")
        coeficientes_modelos = {}

    
    if coeficientes_modelos['regressao_linear'] > 90.0:
        if col3.checkbox('Regr. linear', value=True, key= 'reg_linear'):
            fig.add_trace(go.Scatter(x=acao.index, y=acao['previsao_regressao_linear'], mode='lines', name='Regr. linear', marker_color='rgba(0,255,0,0.5)'))
    
    if coeficientes_modelos['rede_neural'] > 90.0:
        if col4.checkbox('Rede neural', value=True, key = 'neural_net'):
            fig.add_trace(go.Scatter(x=acao.index, y=acao['previsao_rede_neural'], mode='lines', name='Rede neural', marker_color='rgba(0,255,255,0.5)'))
    
    if coeficientes_modelos['hiper_parametro'] > 90.0:
        if col5.checkbox('Hiperparâmetros', value=True, key = 'hyperparam'):
            fig.add_trace(go.Scatter(x=acao.index, y=acao['previsao_rede_neural_hiper_parameter'], mode='lines', name='Hiperparâmetros', marker_color='rgba(255,255,0,0.5)'))
    
    if coeficientes_modelos['random_forest'] > 90.0:    
        if col1.checkbox('Random Forest', value=True, key='rf'):
            fig.add_trace(go.Scatter(
                x=acao.index, y=acao['previsao_random_forest'],
                mode='lines', name='Random Forest', marker_color='rgba(0,100,255,0.5)'
            ))
    if coeficientes_modelos['gradient_boosting'] > 90.0:
        if col2.checkbox('Gradient Boosting', value=True, key='gb'):
            fig.add_trace(go.Scatter(
                x=acao.index, y=acao['previsao_gradient_boosting'],
                mode='lines', name='Gradient Boosting', marker_color='rgba(255,100,0,0.5)'
                ))
    if coeficientes_modelos['svr'] > 90.0:    
        if col3.checkbox('SVR', value=True, key='svr'):
            fig.add_trace(go.Scatter(
                x=acao.index, y=acao['previsao_svr'],
                mode='lines', name='SVR', marker_color='rgba(150,0,255,0.5)'
            ))
    if coeficientes_modelos['ridge'] > 90.0:
        if col4.checkbox('Ridge', value=True, key='ridge'):
            fig.add_trace(go.Scatter(
                x=acao.index, y=acao['previsao_ridge'],
                mode='lines', name='Ridge', marker_color='rgba(0,255,100,0.5)'
            ))
    if coeficientes_modelos['lasso'] > 90.0:
        if col5.checkbox('Lasso', value=True, key='lasso'):
            fig.add_trace(go.Scatter(
                x=acao.index, y=acao['previsao_lasso'],
                mode='lines', name='Lasso', marker_color='rgba(255,0,150,0.5)'
            ))
    
    # Colocar marcador de hoje
    fig.add_trace(go.Scatter(
            x=acao[acao['marcador_hoje'] == 'hoje'].index,
            y=acao[acao['marcador_hoje'] == 'hoje']['Close'],
            mode='markers',
            name='Início previsão',
            marker=dict(color='blue', size=20, symbol='hourglass')
        ))
    
    fig.update_layout(title=f"Gráfico de Preços - {ticker.split('.')[0]}", xaxis_title='Data', yaxis_title='Preço (R$)')
 
    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    st.divider()

def lancar_dataframe(acao, ticker):
    """Lança o DataFrame enriquecido no Streamlit.
    Args:
        acao (pd.DataFrame): DataFrame contendo os dados de preços da ação enriquecidos.
        ticker (str): O ticker da ação a ser analisada.
    """
    # Verifica se o DataFrame está vazio
    if acao.empty:
        st.error("Nenhum dado disponível para o ticker selecionado.")
        return

    # Exibe o DataFrame no Streamlit
    st.dataframe(acao.sort_index(ascending=False))

    # Exibe o DataFrame como CSV para download
    csv = acao.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker.split('.')[0]}_dados_enriquecidos.csv",
        mime='text/csv'
    )

    # Exibindo estatísticas descritivas
    
    st.dataframe(acao.describe())

def mostrar_dados():
    configuracoes_iniciais()
    
    col1, col2 = st.columns(2)
    if col1.button('Importar tickers'):
        importar_tickers()  # Importa os tickers disponíveis
    if col2.button('Atualizar base'):
            atualizar_base_setores()
            
    ticker = definir_ticker()
    if 'Nenhum' in st.session_state.ticker:
        st.error("Por favor, selecione um ticker válido.")
    else:
        fundamentos = importar_fundamentos(ticker)
        if ticker != 'Nenhum':
            mostrar_fundamentos(fundamentos)
        #else:
        #    analise_setor
        acao = baixar_dados(ticker)
        acao = enriquecer_dados(acao)
        st.header("Dados de treino de ML")
        acao = acao_com_preditivo(acao)
        acao = marcador_hoje(acao)
        plotar_grafico(acao, ticker)
        if st.checkbox("Tabelas:"):
            lancar_dataframe(acao, ticker)
        

if __name__ == "__main__":
    configuracoes_iniciais()
    mostrar_dados()