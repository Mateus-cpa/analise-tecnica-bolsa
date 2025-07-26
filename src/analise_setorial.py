import pandas as pd
import streamlit as st
import plotly.express as px

def analise_setorial(df):
    # Gráfico de contagem de tickers por grupo
    grafico_qtd = px.histogram(df, 
                               x="grupo", 
                               title="Contagem de Tickers por Grupo",
                               color="grupo")
    st.plotly_chart(grafico_qtd)

    # Gráfico de variação percentual
    grafico_variacao = px.histogram(df[(df['variacao_valor']>-20) & (df['variacao_valor']<20)], 
                                     x="variacao_valor", 
                                     title="Distribuição da Variação Percentual",
                                     color="grupo")
    st.plotly_chart(grafico_variacao)

    # Gráfico de rendimento
    grafico_rendimento = px.histogram(df[df['rendimento'] <= 30], 
                                       x="rendimento", 
                                       title="Distribuição do Retorno",
                                       color="grupo")
    st.plotly_chart(grafico_rendimento)
    
    # Gráfico de PVP
    grafico_pvp = px.histogram(df[df['pvp'] <= 10], 
                               x="pvp", 
                               title="Distribuição do P/VPA",
                               color="grupo")
    st.plotly_chart(grafico_pvp)
    # utilizar ytdata-profiling para análise mais detalhada
    
