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
        #botão de limpar filtros
        if st.button('Limpar filtros'):
            st.session_state.ticker = 'Nenhum'
            st.session_state.ticker_select = 'Nenhum'
            st.session_state.grupo_select = 'Todos'
            st.session_state.setor_select = 'Todos'
            st.session_state.industrias_select = 'Todos'

        # Lê lista_setores.csv com cabeçalho
        setores_df = pd.read_csv('raw_data/lista_setores_traduzido.csv')  # Espera colunas: ticker, setor, industria

        # Filtro por variação de valor (<-5%, -5% a -1%, -1% a 1%, 1% a 5%, >5%)
        try:
            faixa_variacao = st.segmented_control(
                "Selecione a faixa de variação:",
                options=["<-5%", "-5=>-1%", "-1=>1%", "1=>5%", ">5%"],
                default=None,
                key='faixa_variacao_key'
            )
            # Mapeia as faixas de variação para os valores correspondentes a df_setores['variacao_valor']
            variacao_map = { 
                "<-5%": (min(setores_df['variacao_valor']), -5),
                "-5=>-1%": (-5, -1),
                "-1=>1%": (-1, 1),
                "1=>5%": (1, 5),
                ">5%": (5, max(setores_df['variacao_valor']))
            }
            tupla_variacao = variacao_map[faixa_variacao]
            min_var = tupla_variacao[0]
            max_var = tupla_variacao[1]
            #st.write(f'Variação: {min_var}% a {max_var}%.')
            setores_df = setores_df[(setores_df['variacao_valor'] >= min_var) &
                                    (setores_df['variacao_valor'] <= max_var)]
            setores_df.to_json('bronze_data/setores_filtrados.json', 
                               orient='records', 
                               force_ascii=False)
        except KeyError:
            pass    
        
        # Filtro Rendimento "2%", "5%", "10%", "15%", "20%"
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
            setores_df.to_json('bronze_data/setores_filtrados.json', 
                               orient='records', 
                               force_ascii=False)


        # Filtro por grupo
        grupo = setores_df['grupo'].unique().tolist() # Filtrar por grupo de ticker - Equity (ações), Funds e Index
        grupo_selecionado = st.selectbox('Tipo', options=['Todos'] + grupo, key='grupo_select')
        if grupo_selecionado != 'Todos':
            setores_df = setores_df[setores_df['grupo'] == grupo_selecionado]
            setores_df.to_json('bronze_data/setores_filtrados.json', 
                               orient='records', 
                               force_ascii=False)
        
        # Filtro por setor
        if grupo_selecionado != 'Todos':
            setores_filtrados = setores_df[setores_df['setor_pt'] == grupo_selecionado]['setor_pt'].dropna().unique().tolist()
        else:
            setores_filtrados = setores_df['setor_pt'].unique().tolist() # Filtrar por setor
            setores_df.to_json('bronze_data/setores_filtrados.json',
                                 orient='records',
                                    force_ascii=False)
        setor_selecionado = st.selectbox('Setor', options=['Todos'] + setores_filtrados, key='setor_select')
        
        # Filtro por Indústria
        if grupo_selecionado != 'Todos':
            industrias_filtradas = setores_df[setores_df['grupo'] == grupo_selecionado]['industria_pt'].dropna().unique().tolist()
        elif setor_selecionado != 'Todos': # Filtra subsetores conforme setor
            industrias_filtradas = setores_df[setores_df['setor_pt'] == setor_selecionado]['industria_pt'].dropna().unique().tolist()
        else:
            industrias_filtradas = setores_df['industria_pt'].dropna().unique().tolist()
        industria_selecionada = st.selectbox('Indústria', options=['Todos'] + industrias_filtradas, key='industrias_select')
        # filtra setores_df conforme lista indústrias_selecionadas
        if industria_selecionada != 'Todos':
            setores_df = setores_df[setores_df['industria_pt'] == industria_selecionada]
            setores_df.to_json('bronze_data/setores_filtrados.json', 
                               orient='records', 
                               force_ascii=False)
            
        # Filtragem de tickers pelos dados selecionados
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




