import numpy as np

import pandas as pd
import streamlit as st
import plotly.express as px

def analise_setorial():
    #utilizar a largura da página
    df = st.session_state['setores_filtrados']
    st.warning("Utilize os filtros no menu lateral para encontrar o ticker desejado.")

    st.subheader("Gráficos")    
    colA, colB = st.columns([0.4, 0.6])
    
    dimensao_graficos = colA.radio("Selecione a dimensão dos gráficos:",
                                 ("Valor de Mercado", "Quantidade de Tickers"),
                                 horizontal=True,
                                 key="dimensao_graficos_setorial")

    nome_coluna = colB.radio("Selecione o critério da coluna:",
                            ("Setor", "Indústria"),
                            horizontal=True,
                            key="nome_coluna_setorial")
    coluna = "setor_pt" if nome_coluna == "Setor" else "industria_pt"
    
    col1, col2 = st.columns([0.4, 0.6])
    # Gráficos de contagem ou soma de valor de mercado por grupo
    if dimensao_graficos == "Valor de Mercado":
        grafico_qtd = px.histogram(df, 
                                   x='grupo',
                                   y='valor_mercado',
                                   title="Soma do Valor de Mercado por Grupo",
                                   color="grupo",
                                   histfunc="sum",
                                   hover_data=["ticker", "setor_pt", "industria_pt"])
        grafico_qtd.update_yaxes(title_text="Valor de Mercado (R$)")
    else:
        grafico_qtd = px.histogram(df, 
                                   x='grupo',
                                   title="Contagem de Tickers por Grupo",
                                   color="grupo",
                                   hover_data=["ticker", "setor_pt", "industria_pt"])
        grafico_qtd.update_yaxes(title_text="Contagem")
    grafico_qtd.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    grafico_qtd.update_xaxes(title_text="Grupo")
    grafico_qtd.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_qtd.update_layout(bargap=0.2)
    col1.plotly_chart(grafico_qtd)

    # Gráficos de contagem de tickers por setor/indústria
    if dimensao_graficos == "Valor de Mercado":
        grafico_qtd_por_setor = px.histogram(df, 
                                             y=coluna,
                                             x='valor_mercado',
                                             title=f"Soma do Valor de Mercado por {nome_coluna}",
                                             color="grupo",
                                             histfunc="sum")
        grafico_qtd_por_setor.update_xaxes(title_text="Valor de Mercado (R$)")
    else:
        grafico_qtd_por_setor = px.histogram(df, 
                                             y=coluna,
                                             title=f"Contagem de Tickers por {nome_coluna}",
                                             color="grupo")
        grafico_qtd_por_setor.update_xaxes(title_text="Contagem")
    grafico_qtd_por_setor.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_qtd_por_setor.update_yaxes(title_text=nome_coluna)
    grafico_qtd_por_setor.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_qtd_por_setor.update_layout(bargap=0.2)
    col2.plotly_chart(grafico_qtd_por_setor)
    
    # Gráficos de variação percentual
    if dimensao_graficos == "Valor de Mercado":
        grafico_variacao = px.histogram(df[(df['variacao_valor']>-20) & (df['variacao_valor']<20)], 
                                         x="variacao_valor", 
                                         y="valor_mercado",
                                         title="Soma do Valor de Mercado por Variação Percentual",
                                         color="grupo",
                                         labels={"variacao_valor": "Variação Percentual"},
                                         histfunc="sum")
        grafico_variacao.update_yaxes(title_text="Valor de Mercado (R$)")
    else:
        grafico_variacao = px.histogram(df[(df['variacao_valor']>-20) & (df['variacao_valor']<20)], 
                                         x="variacao_valor", 
                                         title="Distribuição da Variação Percentual",
                                         color="grupo",
                                         labels={"variacao_valor": "Variação Percentual"})
        grafico_variacao.update_yaxes(title_text="Contagem")
    grafico_variacao.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_variacao.update_xaxes(title_text="Variação Percentual")
    grafico_variacao.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_variacao.update_layout(bargap=0.2)
    col1.plotly_chart(grafico_variacao)

    boxplot_variacao = px.box((df[(df['variacao_valor']<50) & 
                                  (df['variacao_valor']>-20)]),
                            y=coluna, 
                            x="variacao_valor",
                            title=f"Boxplot da Variação Percentual por {nome_coluna}",
                            #points="all",
                            color="grupo")
    boxplot_variacao.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    boxplot_variacao.update_yaxes(title_text=nome_coluna)
    boxplot_variacao.update_xaxes(title_text="Variação Percentual")
    boxplot_variacao.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    boxplot_variacao.update_layout(bargap=0.2)  # Ajusta
    col2.plotly_chart(boxplot_variacao)

    # Gráficos de rendimento
    if dimensao_graficos == "Valor de Mercado":
        grafico_rendimento = px.histogram(df[df['rendimento'] <= 30], 
                                           x="rendimento", 
                                           y="valor_mercado",
                                           title="Soma do Valor de Mercado por Retorno",
                                           color="grupo",
                                           histfunc="sum")
        grafico_rendimento.update_yaxes(title_text="Valor de Mercado (R$)")
    else:
        grafico_rendimento = px.histogram(df[df['rendimento'] <= 30], 
                                           x="rendimento", 
                                           title="Distribuição do Retorno",
                                           color="grupo")
        grafico_rendimento.update_yaxes(title_text="Contagem")
    grafico_rendimento.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    grafico_rendimento.update_xaxes(title_text="Rendimento (%)")
    grafico_rendimento.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_rendimento.update_layout(bargap=0.2)
    col1.plotly_chart(grafico_rendimento)

    boxplot_rendimento = px.box(df[(df['rendimento']<30) & (df['rendimento']>0)],
                                y=coluna,
                                x="rendimento",
                                title=f"Boxplot do Rendimento por {nome_coluna}",
                                color="grupo")
    boxplot_rendimento.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    boxplot_rendimento.update_yaxes(title_text=nome_coluna)
    boxplot_rendimento.update_xaxes(title_text="Rendimento (%)")
    boxplot_rendimento.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    boxplot_rendimento.update_layout(bargap=0.2)  # Ajusta
    col2.plotly_chart(boxplot_rendimento)

    # Gráficos de PVP
    if dimensao_graficos == "Valor de Mercado":
        grafico_pvp = px.histogram(df[df['pvp'] <= 10], 
                                   x="pvp", 
                                   y="valor_mercado",
                                   title="Soma do Valor de Mercado por P/VPA",
                                   color="grupo",
                                   histfunc="sum")
        grafico_pvp.update_yaxes(title_text="Valor de Mercado (R$)")
    else:
        grafico_pvp = px.histogram(df[df['pvp'] <= 10], 
                                   x="pvp", 
                                   title="Distribuição do P/VPA",
                                   color="grupo")
        grafico_pvp.update_yaxes(title_text="Contagem")
    grafico_pvp.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    grafico_pvp.update_xaxes(title_text="P/VPA")
    grafico_pvp.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_pvp.update_layout(bargap=0.2)
    col1.plotly_chart(grafico_pvp)

    boxplot_pvp = px.box(df[(df['pvp']<10) & (df['pvp']>0)],
                         y=coluna,
                         x="pvp",
                         title=f"Boxplot do P/VPA por {nome_coluna}",
                         color="grupo")
    boxplot_pvp.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    boxplot_pvp.update_yaxes(title_text=nome_coluna)
    boxplot_pvp.update_xaxes(title_text="P/VPA")
    boxplot_pvp.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    boxplot_pvp.update_layout(bargap=0.2)  # Ajusta
    col2.plotly_chart(boxplot_pvp)

    # listar tikers como botões para filtrar
    tickers = df['ticker'].unique().tolist()
    st.subheader(f"Lista dos {len(tickers)} tickers")
    tickers.sort()
    if len(tickers) < 40:
        ticker_columns = np.array_split(tickers, 5)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                for ticker in ticker_columns[i]:
                    if st.button(ticker):
                        st.session_state.ticker = f'{ticker}.SA'

    else:
        st.warning(f"Existem {len(tickers)} tickers disponíveis. Use os filtros para encontrar o ticker desejado.")
    # utilizar ytdata-profiling para análise mais detalhada
    
