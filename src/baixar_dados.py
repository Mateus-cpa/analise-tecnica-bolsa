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
        setores_df = pd.read_csv('raw_data/lista_setores_traduzido.csv')  # Espera colunas: ticker, setor, industria

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
                key='variacao_slider')
            setores_df = setores_df[
                (setores_df['variacao_valor'] >= faixa_variacao[0]) &
                (setores_df['variacao_valor'] <= faixa_variacao[1])]                             

        # Filtro condicional por rendimento
        aplicar_filtro_rendimento = st.checkbox('Filtrar por Dividend Yield')
        if aplicar_filtro_rendimento and 'rendimento' in setores_df.columns:
            dy_minino = float(st.text_input('Informe o Rendimento anual (DY) mínimo'))
            setores_df = setores_df[setores_df['rendimento'] >= dy_minino]
        
        # Filtro por grupo
        grupo = setores_df['grupo'].unique().tolist() # Filtrar por grupo de ticker - Equity (ações), Funds e Index
        grupo_selecionado = st.selectbox('Tipo', options=['Todos'] + grupo, key='grupo_select')
        
        # Filtro por setor
        if grupo_selecionado != 'Todos':
            setores_filtrados = setores_df[setores_df['setor_pt'] == grupo_selecionado]['setor_pt'].dropna().unique().tolist()
        else:
            setores_filtrados = setores_df['setor_pt'].unique().tolist() # Filtrar por setor
        setor_selecionado = st.selectbox('Setor', options=['Todos'] + setores_filtrados, key='setor_select')
        
        # Filtro por Indústria
        if setor_selecionado != 'Todos': # Filtra subsetores conforme setor
            industrias_filtradas = setores_df[setores_df['setor_pt'] == setor_selecionado]['industria_pt'].dropna().unique().tolist()
        else:
            industrias_filtradas = setores_df['industria_pt'].dropna().unique().tolist()
        industria_selecionada = st.selectbox('Indústria', options=['Todos'] + industrias_filtradas, key='industrias_select')

        
        # Segunda linha de filtros por nome ou ticker
        # Filtragem apenas pelos selects
        setores_filtrados_df = setores_df.copy()
        if grupo_selecionado != 'Todos':
            setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['grupo'] == grupo_selecionado]
        if setor_selecionado != 'Todos':
            setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['setor_pt'] == setor_selecionado]
        if industria_selecionada != 'Todos':
            setores_filtrados_df = setores_filtrados_df[setores_filtrados_df['industria_pt'] == industria_selecionada]

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




