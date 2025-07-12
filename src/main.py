#bibliotecas nativas
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json

# bibliotecas de terceiros
#from talib import RSI # Technical Analysis - TA-Lib
import streamlit as st # Streamlit para interface web
#from dateutil.relativedelta import relativedelta
#from pandas.tseries.offsets import BDay

#bibliotecas locais
from importar_tickers import importar_tickers # Importando a função para definir o ticker
from baixar_dados import baixar_dados, definir_ticker
from importar_fundamentos import importar_fundamentos # Importando a função para importar fundamentos
from atualizar_base_setores import atualizar_base_setores
from modelo_preditivo import acao_com_preditivo
from tratamento_ativo import enriquecer_dados, detectar_mudanca_tendencia, marcador_hoje, adicionar_target_median_price
from plotar_grafico import plotar_grafico
from mostrar_fundamentos import mostrar_fundamentos

def configuracoes_iniciais():
    # Configurações iniciais
    plt.style.use('dark_background')  # Corrigido: 'darkgrid' não existe, use 'dark_background'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)



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

def tela_streamlit():
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
        with st.sidebar:
            tempo_anos = st.selectbox(label='Qtde. de anos de download', options=range(20, 0, -1))
        acao = baixar_dados(ticker, tempo_anos)
        acao = enriquecer_dados(acao)
        try:
            acao = acao_com_preditivo(acao) #dados ML
        except ValueError:
            st.warning('Não foram calculados dados de previsão com Machine Learning.')
            with open('bronze_data/coeficientes_modelos.json', mode='w') as coef_file:
                file_coef = {
                    "regressao_linear": 0.0,
                    "rede_neural": 0.0,
                    "hiper_parametro": 0.0,
                    "random_forest": 0.0,
                    "gradient_boosting": 0.0,
                    "svr": 0.0,
                    "ridge": 0.0,
                    "lasso": 0.0
                }
                json.dump(file_coef, coef_file, indent=4)  # Esta linha grava o dicionário no arquivo

        acao = marcador_hoje(acao)
        # Obtém o targetMedianPrice do DataFrame fundamentos
        target_median_price = fundamentos['targetMedianPrice'].iloc[0] if 'targetMedianPrice' in fundamentos.columns else None
        acao = adicionar_target_median_price(acao=acao,
                                             target_median_price=target_median_price)
        plotar_grafico(acao, ticker)
        if st.checkbox("Tabelas:"):
            lancar_dataframe(acao, ticker)
        

if __name__ == "__main__":
    tela_streamlit()