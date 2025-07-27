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

    # Gráfico de contagem de tickers por setor/indústria
    coluna_qtd = col2.radio("Selecione a coluna:", 
                            ("setor_pt", "industria_pt"),
                            horizontal=True)
    nome_coluna_qtd = "Setor" if coluna_qtd == "setor_pt" else "Indústria"
    grafico_qtd_por_setor = px.histogram(df, 
                                         y=coluna_qtd,
                                         title= f"Contagem de Tickers por {nome_coluna_qtd}",
                                         color="grupo"
                                         )
    grafico_qtd_por_setor.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    grafico_qtd_por_setor.update_yaxes(title_text=nome_coluna_qtd)
    grafico_qtd_por_setor.update_xaxes(title_text="Contagem")
    col2.plotly_chart(grafico_qtd_por_setor)
    st.divider()
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
    
