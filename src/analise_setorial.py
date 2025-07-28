import pandas as pd
import streamlit as st
import plotly.express as px

def analise_setorial():
    #utilizar a largura da página
    df = pd.read_json('bronze_data/setores_filtrados.json', orient='records')

    nome_coluna = st.radio("Selecione a coluna:", 
                            ("Setor", "Indústria"),
                            horizontal=True)
    coluna = "setor_pt" if nome_coluna == "Setor" else "industria_pt"
    
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
    grafico_qtd.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_qtd.update_layout(bargap=0.2)  # Ajusta o espaçamento entre as barras
    col1.plotly_chart(grafico_qtd)

    # Gráfico de contagem de tickers por setor/indústria
    grafico_qtd_por_setor = px.histogram(df, 
                                         y=coluna,
                                         title= f"Contagem de Tickers por {nome_coluna}",
                                         color="grupo"
                                         )
    grafico_qtd_por_setor.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_qtd_por_setor.update_yaxes(title_text=nome_coluna)
    grafico_qtd_por_setor.update_xaxes(title_text="Contagem")
    grafico_qtd_por_setor.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_qtd_por_setor.update_layout(bargap=0.2)
    col2.plotly_chart(grafico_qtd_por_setor)
    
    # Gráfico de variação percentual com leganda abaixo do gráfico
    grafico_variacao = px.histogram(df[(df['variacao_valor']>-20) & (df['variacao_valor']<20)], 
                                     x="variacao_valor", 
                                     title="Distribuição da Variação Percentual",
                                     color="grupo",
                                     labels={"variacao_valor": "Variação Percentual"})
    grafico_variacao.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_variacao.update_xaxes(title_text="Variação Percentual")
    grafico_variacao.update_yaxes(title_text="Contagem")
    grafico_variacao.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_variacao.update_layout(bargap=0.2)  # Ajusta
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

    # Gráfico de PVP
    grafico_pvp = px.histogram(df[df['pvp'] <= 10], 
                               x="pvp", 
                               title="Distribuição do P/VPA",
                               color="grupo")
    grafico_pvp.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    grafico_pvp.update_xaxes(title_text="P/VPA")
    grafico_pvp.update_yaxes(title_text="Contagem")
    grafico_pvp.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    grafico_pvp.update_layout(bargap=0.2)  # Ajusta
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

    # listar tikers como botões para filtrar
    st.subheader("Lista de Tickers")
    tickers = df['ticker'].unique().tolist()
    tickers.sort()
    if len(tickers) < 40:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:  
            for ticker in tickers[:len(tickers)//5]:
                if st.button(ticker):
                    st.session_state.ticker = ticker
        with col2:
            for ticker in tickers[len(tickers)//5:len(tickers)//5*2]:
                if st.button(ticker):
                    st.session_state.ticker = ticker
        with col3:
            for ticker in tickers[len(tickers)//5*2:len(tickers)//5*3]:
                if st.button(ticker):
                    st.session_state.ticker = ticker
        with col4:
            for ticker in tickers[len(tickers)//5*3:len(tickers)//5*4]:
                if st.button(ticker):
                    st.session_state.ticker = ticker
        with col5:
            for ticker in tickers[len(tickers)//5*4:]:
                if st.button(ticker):
                    st.session_state.ticker = ticker
    # utilizar ytdata-profiling para análise mais detalhada
    
