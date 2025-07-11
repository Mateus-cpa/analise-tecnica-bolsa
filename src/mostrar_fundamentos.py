import pandas as pd
import streamlit as st

def mostrar_fundamentos(fundamentos: pd.DataFrame):
    """Mostra os fundamentos da ação no Streamlit.
    Args:
        fundamentos (pd.DataFrame): DataFrame contendo os fundamentos da ação.
    """
    st.header(f"{fundamentos['shortName'].values[0]} - {fundamentos['symbol'].values[0]}")
    st.write(f"Nome completo: {fundamentos['longName'].values[0]}")
    if fundamentos['sector'].values[0] == None:
        st.write(f"Setor: {fundamentos['sector'].values[0]}")
    if fundamentos['industry'].values[0] == None:
        st.write(f"Indústria: {fundamentos['industry'].values[0]}")

    st.subheader("Fundamentos")
    if fundamentos.empty:
        st.error("Nenhum dado fundamental disponível para o ticker selecionado.")
        return
    # Exibe os fundamentos no Streamlit
    col1, col2, col3 = st.columns(3)
    with col1:
        #kpis
        if fundamentos['previousClose'].values[0] is not None:
            variacao = fundamentos['regularMarketChangePercent'].values[0]
            st.metric(
                "Último Fechamento",
                f"R$ {fundamentos['previousClose'].values[0]:.2f}",
                delta=f"{variacao:.2f}%",
                delta_color="normal"
            )
        dy = fundamentos['dividendYield'].values[0]
        if pd.notnull(dy) and dy != 'N/A':
            st.metric("Dividend Yield", f"{float(dy)/100:.2%}")
        if fundamentos['lastDividendValue'].values[0] is not None:
            st.metric("Último Dividendo", f"R$ {fundamentos['lastDividendValue'].values[0]:.2f}")
        last_div_date = fundamentos['lastDividendDate'].values[0]
        if pd.notnull(last_div_date):
            try:
                data_formatada = pd.to_datetime(last_div_date, unit='s').strftime('%d/%m/%Y')
                st.metric("Data do Último Dividendo", data_formatada)
            except Exception as e:
                st.error(f"Erro ao exibir a data do último dividendo: {e}")
        else:
            st.metric("Data do Último Dividendo", "N/A")

    with col2:
        if fundamentos['profitMargins'].values[0] is not None:
            st.metric("Margem de Lucro", f"{fundamentos['profitMargins'].values[0]:.2%}")
        if fundamentos['recommendationKey'].values[0] is not None:
            if 'buy' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Comprar", delta_color="normal")
                #st.metric("Recomendação", ":cow: [Comprar]", delta_color="normal")
            elif 'sell' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Vender", delta_color="inverse")
                #st.metric("Recomendação", ":bear: [Vender]", delta_color="inverse")
        if fundamentos['numberOfAnalystOpinions'].values[0] is not None:
            st.metric('Nº de Opiniões de Analistas', fundamentos['numberOfAnalystOpinions'].values[0])
        if fundamentos['targetMedianPrice'].values[0] is not None:
            st.metric('Mediana de Preço alvo', f"R$ {fundamentos['targetMedianPrice'].values[0]:.2f}")

    with col3:
        if fundamentos['targetHighPrice'].values[0] is not None:
            st.metric('Preço alvo máximo', f"R$ {fundamentos['targetHighPrice'].values[0]:.2f}")
        if fundamentos['targetLowPrice'].values[0] is not None:
            st.metric('Preço alvo mínimo', f"R$ {fundamentos['targetLowPrice'].values[0]:.2f}")
        if fundamentos['targetMeanPrice'].values[0] is not None:
            st.metric('Média de Preço alvo', f"R$ {fundamentos['targetMeanPrice'].values[0]:.2f}")

    mostrar_fundamentos = st.checkbox('Mostrar todos fundamentos')
    if mostrar_fundamentos:
        for col in fundamentos.columns:
            st.write(f"{col}: {fundamentos[col].values[0]}")