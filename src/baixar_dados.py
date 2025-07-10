from datetime import date, timedelta
import pandas as pd

import yfinance as yf
import streamlit as st

@st.cache_data(show_spinner="Baixando dados...", max_entries=10)
def baixar_dados(ticker=None, tempo_anos=19):
    """Baixa os dados do Yahoo Finance para o ticker especificado.
    Args:
        ticker (str): O ticker da ação a ser analisada.
        tempo_anos (int): O número de anos de dados a serem baixados.
    Returns:
        pd.DataFrame: DataFrame contendo os dados de preços da ação.
    """
    if not ticker or ticker == 'Nenhum':
        return pd.DataFrame()
    start_date = date.today() - timedelta(days=tempo_anos*365)
    end_date = date.today()
    intervalo = '1d'
    df = yf.download(ticker, start=start_date, end=end_date, interval=intervalo)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    return df

def definir_ticker():
    """Define o ticker a ser analisado.
    Returns:
        str: O ticker da ação a ser analisada.
    """
    with st.sidebar:
        st.header("Definir Ticker")
    
        # Lê lista_setores.csv com cabeçalho
        setores_df = pd.read_csv('raw_data/lista_setores.csv')  # Espera colunas: ticker, setor, industria

        # Filtro condicional por variação
        aplicar_filtro_variacao = st.checkbox('Filtrar por variação de valor de mercado')
        if aplicar_filtro_variacao and 'variacao_valor' in setores_df.columns:
            min_var, max_var = float(setores_df['variacao_valor'].min()), float(setores_df['variacao_valor'].max())
            faixa_variacao = st.slider(
                'Variação (%)',
                min_value=min_var,
                max_value=max_var,
                value=(min_var, max_var),
                step=0.1,
                key='variacao_slider'
            )
            setores_df = setores_df[
                (setores_df['variacao_valor'] >= faixa_variacao[0]) &
                (setores_df['variacao_valor'] <= faixa_variacao[1])
            ]                             

        # Filtro por grupo
        grupo = setores_df['grupo'].unique().tolist() # Filtrar por grupo de ticker - Equity (ações), Funds e Index
        grupo_selecionado = st.selectbox('Tipo', options=['Todos'] + grupo, key='grupo_select')
        
        # Filtro por setor
        if grupo_selecionado != 'Todos':
            setores_filtrados = setores_df[setores_df['setor'] == grupo_selecionado]['setor'].dropna().unique().tolist()
        else:
            setores_filtrados = setores_df['setor'].unique().tolist() # Filtrar por setor
        setor_selecionado = st.selectbox('Setor', options=['Todos'] + setores_filtrados, key='setor_select')
        
        # Filtro por Indústria
        if setor_selecionado != 'Todos': # Filtra subsetores conforme setor
            industrias_filtradas = setores_df[setores_df['setor'] == setor_selecionado]['industria'].dropna().unique().tolist()
        else:
            industrias_filtradas = setores_df['industria'].dropna().unique().tolist()
        industria_selecionada = st.selectbox('Indústria', options=['Todos'] + industrias_filtradas, key='industrias_select')

        
        # Segunda linha de filtros por nome ou ticker
        # Filtragem apenas pelos selects
        setores_filtrados_df = setores_df.copy()
        if grupo_selecionado != 'Todos':
            setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['grupo'] == grupo_selecionado]
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
            variacao = fundamentos['regularMarketChangePercent'].values[0]
            st.metric(
                "Último Fechamento",
                f"R$ {fundamentos['previousClose'].values[0]:.2f}",
                delta=f"{variacao:.2f}%",
                delta_color="normal"
            )
        if fundamentos['dividendYield'].values[0] is not None:
            st.metric("Dividend Yield", f"{fundamentos['dividendYield'].values[0]/100:.2%}")
        if fundamentos['lastDividendValue'].values[0] is not None:
            st.metric("Último Dividendo", f"R$ {fundamentos['lastDividendValue'].values[0]:.2f}")
        try:
            st.metric("Data do Último Dividendo", f"{pd.to_datetime(fundamentos['lastDividendDate'].values[0], unit='s').strftime('%d/%m/%Y')}")
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


