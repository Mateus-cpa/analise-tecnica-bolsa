from datetime import timedelta
import json

import pandas as pd
import streamlit as st
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

    with st.sidebar: #seleciona dia inicial
        data_inicio = st.date_input(
            "Selecione a data de início",
            value=acao.index.max().date() - timedelta(days=67),
            min_value=acao.index.min().date(),
            max_value=acao.index.max().date(),
            key="data_inicio_calendario"
        )
    data_fim = acao.index.max().date()

    # Filtra o DataFrame conforme o período selecionado
    acao = acao.loc[(acao.index.date >= data_inicio) & (acao.index.date <= data_fim)]

    
    # Plotando o gráfico de preços
    fig = make_subplots(rows=1, cols=1)
    
    # Adicionando elementos no gráfico
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.checkbox('MM5', value=False):
        fig.add_trace(go.Scatter(x=acao.index, y=acao['MM5'], mode='lines', name='MM5', marker_color='rgba(250,250,250,0.5)'))
    if col1.checkbox('Candles', value=True):
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
    
    # Adiciona ponto diamante para targetMedianPrice na última linha, se existir
    if 'targetMedianPrice' in acao.columns and pd.notnull(acao['targetMedianPrice'].iloc[-1]):
        fig.add_trace(go.Scatter(
            x=[acao.index[-1]],
            y=[acao['targetMedianPrice'].iloc[-1]],
            mode='markers',
            name='Previsão dos analistas',
            marker=dict(color='purple', size=10, symbol='diamond')
        ))
    
    # Adiciona seta do último fechamento para o targetMedianPrice
        
    
    fig.update_layout(title=f"Gráfico de Preços - {ticker.split('.')[0]}", xaxis_title='Data', yaxis_title='Preço (R$)')
 
    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    st.divider()

if __name__ == 'main':
    plotar_grafico()