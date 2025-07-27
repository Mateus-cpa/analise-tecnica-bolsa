import pandas as pd
import streamlit as st
import plotly.express as px

def analise_setorial():
    #utilizar a largura da página
    df = pd.read_json('bronze_data/setores_filtrados.json', orient='records')

    col1, col2 = st.columns([0.4, 0.6])
    
    # Gráfico de contagem de tickers por grupo
    grafico_qtd = px.histogram(df, 
                               x="grupo", 
                               title="Contagem de Tickers por Grupo",
                               color="grupo",
                               #mortar ao apontar no gráfico os tickers
                               hover_data=["ticker", "setor_pt", "industria_pt"]
                               )
    grafico_qtd.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    grafico_qtd.update_xaxes(title_text="Grupo")
    grafico_qtd.update_yaxes(title_text="Contagem")
    col1.plotly_chart(grafico_qtd)
    grafico_qtd_por_setor = px.histogram(df, 
                                         y="setor_pt",
                                         title="Contagem de Tickers por Setor",
                                         color="grupo",
                                         labels={"setor_pt": "Setor","count": "Contagem"}
                                         )
    grafico_qtd_por_setor.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_qtd_por_setor.update_yaxes(title_text="Setor")
    grafico_qtd_por_setor.update_xaxes(title_text="Contagem")
    col2.plotly_chart(grafico_qtd_por_setor)

    # Gráfico de variação percentual com leganda abaixo do gráfico
    grafico_variacao = px.histogram(df[(df['variacao_valor']>-20) & (df['variacao_valor']<20)], 
                                     x="variacao_valor", 
                                     title="Distribuição da Variação Percentual",
                                     color="grupo",
                                     labels={"variacao_valor": "Variação Percentual"})
    grafico_variacao.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_variacao.update_xaxes(title_text="Variação Percentual")
    col1.plotly_chart(grafico_variacao)

    boxplot_variacao = px.box((df[(df['variacao_valor']<50) & 
                                  (df['variacao_valor']>-20)]),
                            y="setor_pt", 
                            x="variacao_valor",
                            title="Boxplot da Variação Percentual por Setor",
                            #points="all",
                            color="grupo")
    boxplot_variacao.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    boxplot_variacao.update_yaxes(title_text="Setor")
    boxplot_variacao.update_xaxes(title_text="Variação Percentual")
    col2.plotly_chart(boxplot_variacao)

    # Gráfico de rendimento
    grafico_rendimento = px.histogram(df[df['rendimento'] <= 30], 
                                       x="rendimento", 
                                       title="Distribuição do Retorno",
                                       color="grupo")
    grafico_rendimento.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    grafico_rendimento.update_xaxes(title_text="Rendimento (%)")
    grafico_rendimento.update_yaxes(title_text="Contagem")
    grafico_rendimento.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_rendimento.update_layout(bargap=0.2)  # Ajusta
    col1.plotly_chart(grafico_rendimento)

    boxplot_rendimento = px.box(df[(df['rendimento']<30) & (df['rendimento']>0)],
                                y="setor_pt",
                                x="rendimento",
                                title="Boxplot do Rendimento por Setor",
                                color="grupo")
    boxplot_rendimento.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75))
    boxplot_rendimento.update_yaxes(title_text="Setor")
    boxplot_rendimento.update_xaxes(title_text="Rendimento (%)")
    col2.plotly_chart(boxplot_rendimento)

    # Gráfico de PVP
    grafico_pvp = px.histogram(df[df['pvp'] <= 10], 
                               x="pvp", 
                               title="Distribuição do P/VPA",
                               color="grupo")
    st.plotly_chart(grafico_pvp)
    # utilizar ytdata-profiling para análise mais detalhada
    
