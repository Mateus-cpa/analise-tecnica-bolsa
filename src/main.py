#bibliotecas nativas
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json
import sys

# bibliotecas de terceiros
import streamlit as st # Streamlit para interface web

#bibliotecas locais
from importar_tickers import importar_tickers # Importando a função para definir o ticker
from painel_lateral import baixar_dados, definir_ticker
from importar_fundamentos import importar_fundamentos # Importando a função para importar fundamentos
from atualizar_base_setores import atualizar_base_setores
from modelo_preditivo import acao_com_preditivo
from tratamento_ativo import enriquecer_dados, marcador_hoje, adicionar_target_median_price
from plotar_grafico import plotar_grafico
from mostrar_fundamentos import mostrar_fundamentos
from analise_setorial import analise_setorial

def configuracoes_iniciais():
    # Configurações iniciais
    plt.style.use('dark_background')  # Corrigido: 'darkgrid' não existe, use 'dark_background'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    st.set_page_config(layout="wide")
    if 'ticker' not in st.session_state:
        # Inicializa o ticker como 'NENHUM' se não estiver definido
        st.session_state.ticker = 'NENHUM'



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
    
        #importação
    col1, col2 = st.columns(2)
    if col1.button('Importar tickers'):
        importar_tickers()  # Importa os tickers disponíveis
    if col2.button('Atualizar base'):
            atualizar_base_setores()

    if 'setores_filtrados' not in st.session_state:
        with open('raw_data/lista_setores_traduzido.csv', 'r', encoding='utf-8') as f:
            st.session_state['setores_filtrados'] = pd.read_csv(f)
    if ('ticker' not in st.session_state) or (st.session_state.ticker is None) or (st.session_state.ticker == 'NENHUM'):
        st.session_state.ticker = definir_ticker()
    if (st.session_state.ticker is None or st.session_state.ticker == 'NENHUM'):
        st.header(" Análise Setorial")
        st.session_state.ticker = analise_setorial()
        
    
    if st.session_state.ticker != 'NENHUM':
        fundamentos = importar_fundamentos(st.session_state.ticker)
        mostrar_fundamentos(fundamentos)
        
        with st.sidebar:
            tempo_anos = st.selectbox(label='Qtde. de anos de download', options=range(20, 0, -1))
        acao = baixar_dados(st.session_state.ticker, tempo_anos)
        
        
        try:
            acao = enriquecer_dados(acao)
        except IndexError:
            st.warning('Não foi possível calcular indicadores.')
        try:
            acao = acao_com_preditivo(acao) #dados ML
        except KeyError:
            st.warning('Não foram calculados dados de previsão com Machine Learning.')
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
        plotar_grafico(acao, st.session_state.ticker)
        if st.checkbox("Histórico do ativo"):
            lancar_dataframe(acao, st.session_state.ticker)
    st.subheader("Base de dados de setores")
    # colocar seção retrátil st.expander
    with st.expander("Ver setores disponíveis"):
        with open('raw_data/lista_setores_traduzido.csv', 'r', encoding='utf-8') as f:
            setores_df = pd.read_csv(f)
            if st.button("Baixar setores"):
                setores_df.to_csv('bronze_data/setores_filtrados.json', orient='records', index=False)
                st.success("Setores baixados com sucesso!")
        st.dataframe(setores_df)
            
    st.write(f"Versão do python: {str(sys.version).split('(')[0]}")


if __name__ == "__main__":
    tela_streamlit()