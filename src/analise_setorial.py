import pandas as pd
import streamlit as st
import plotly.express as px

def analise_setorial(df):
    # Gráfico de contagem de tickers por grupo
    grafico_qtd = px.histogram(df, x="grupo", title="Contagem de Tickers por Grupo")
    st.plotly_chart(grafico_qtd)
    grafico_variacao = px.histogram(df, x="variacao_valor", title="Distribuição da Variação Percentual")
    st.plotly_chart(grafico_variacao)
    grafico_rendimento = px.histogram(df, x="rendimento", title="Distribuição do Retorno")
    st.plotly_chart(grafico_rendimento)