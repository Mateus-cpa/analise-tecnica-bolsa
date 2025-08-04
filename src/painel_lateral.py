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
        setores_df = pd.read_csv('raw_data/lista_setores_traduzido.csv')

        # -- LIMPAR FILTROS --
        if st.button('Limpar filtros'):
            st.session_state.ticker = 'NENHUM'
            st.session_state.ticker_select = 'Nenhum'
            st.session_state.grupo_select = 'Todos'
            st.session_state.setor_select = 'Todos'
            st.session_state.industrias_select = 'Todos'
            st.session_state['setores_filtrados'] = setores_df.copy()
            #limpar faixa de variação e rendimento
            st.session_state['faixa_variacao_key'] = None
            st.session_state['minimo_dy'] = None

        # -- VARIAÇÃO DE VALOR --
        try:
            faixa_variacao = st.segmented_control(
                "Selecione a faixa de variação:",
                options=["<-5%", "-5=>-1%", "-1=>1%", "1=>5%", ">5%"],
                default=None,
                key='faixa_variacao_key'
            )
            variacao_map = { 
                "<-5%": (min(setores_df['variacao_valor']), -5),
                "-5=>-1%": (-5, -1),
                "-1=>1%": (-1, 1),
                "1=>5%": (1, 5),
                ">5%": (5, float('inf'))

            }
            tupla_variacao = variacao_map[faixa_variacao]
            min_var = tupla_variacao[0]
            max_var = tupla_variacao[1]
            if faixa_variacao == ">5%":
                setores_df = setores_df[setores_df['variacao_valor'] > 5]
            else:
                setores_df = setores_df[(setores_df['variacao_valor'] >= min_var) &
                                        (setores_df['variacao_valor'] <= max_var)]
            st.session_state['setores_filtrados'] = setores_df.copy()
        except KeyError:
            pass    

        # -- RENDIMENTO (DY) --
        try:
            minimo_dy = st.segmented_control(
                "DY mínimo:",
                options=["2%", "5%", "10%", "15%", "20%"],
                default=None,
                key='minimo_dy'
            )
        except TypeError:
            pass
        if minimo_dy:
            minimo_dy_float = float(minimo_dy.replace('%', ''))
            setores_df = setores_df[setores_df['rendimento'] >= minimo_dy_float]
            st.session_state['setores_filtrados'] = setores_df.copy()

        # -- GRUPO --
        grupo = st.session_state['setores_filtrados']['grupo'].unique().tolist()
        grupo_selecionado = st.selectbox('Tipo', options=['Todos'] + grupo, key='grupo_select')
        if grupo_selecionado != 'Todos':
            st.session_state['setores_filtrados'] = st.session_state['setores_filtrados'][st.session_state['setores_filtrados']['grupo'] == grupo_selecionado]

        # -- SETOR --
        if grupo_selecionado != 'Todos':
            setores_filtrados = st.session_state['setores_filtrados']['setor_pt'].dropna().unique().tolist()
        else:
            setores_filtrados = st.session_state['setores_filtrados']['setor_pt'].unique().tolist()
        setor_selecionado = st.selectbox('Setor', options=['Todos'] + setores_filtrados, key='setor_select')
        if setor_selecionado != 'Todos':
            st.session_state['setores_filtrados'] = st.session_state['setores_filtrados'][st.session_state['setores_filtrados']['setor_pt'] == setor_selecionado]

        # -- INDÚSTRIA --
        if setor_selecionado != 'Todos':
            industrias_filtradas = st.session_state['setores_filtrados']['industria_pt'].dropna().unique().tolist()
        else:
            industrias_filtradas = st.session_state['setores_filtrados']['industria_pt'].dropna().unique().tolist()
        industria_selecionada = st.selectbox('Indústria', options=['Todos'] + industrias_filtradas, key='industrias_select')
        if industria_selecionada != 'Todos':
            st.session_state['setores_filtrados'] = st.session_state['setores_filtrados'][st.session_state['setores_filtrados']['industria_pt'] == industria_selecionada]

        # -- TICKER --
        setores_filtrados_df = st.session_state['setores_filtrados'].copy()
        setores_filtrados_df['ticker_busca'] = setores_filtrados_df.apply(
            lambda linha: f"{str(linha['ticker'])} {str(linha.get('nome',''))} {str(linha.get('nome completo',''))}", axis=1
        )
        opcoes_dict = {row['ticker_busca']: row['ticker'] for _, row in setores_filtrados_df.iterrows()}
        opcoes_lista = ['Nenhum'] + list(opcoes_dict.keys())

        selecionado = st.selectbox(
            "Ticker da ação",
            options=opcoes_lista,
            key="ticker_select"
        )
        st.session_state.ticker = opcoes_dict.get(selecionado, 'Nenhum')

        ticker = st.session_state.ticker + '.SA' if st.session_state.ticker != 'Nenhum' else 'Nenhum'

        # -- Mostrar filtros aplicados --
        st.sidebar.markdown("## Filtros Aplicados")
        st.sidebar.write(f"**Variação de Valor:** {faixa_variacao if 'faixa_variacao_key' in st.session_state else 'Nenhum'}")
        st.sidebar.write(f"**Rendimento (DY):** {minimo_dy if 'minimo_dy' in st.session_state else 'Nenhum'}")
        st.sidebar.write(f"**Grupo:** {grupo_selecionado if 'grupo_select' in st.session_state else 'Todos'}")
        st.sidebar.write(f"**Setor:** {setor_selecionado if 'setor_select' in st.session_state else 'Todos'}")
        st.sidebar.write(f"**Indústria:** {industria_selecionada if 'industrias_select' in st.session_state else 'Todos'}") 

        # -- Retorna o ticker selecionado --
        return ticker.upper()



