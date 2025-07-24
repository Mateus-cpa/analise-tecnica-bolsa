import pandas as pd
import streamlit as st
from googletrans import Translator

def translate_text(text_to_translate: str, dest_lang: str = 'pt') -> str:
    """
    Translates the given text asynchronously to the destination language.

    Args:
        text_to_translate (str): The text to be translated.
        dest_lang (str): The destination language code (default is 'pt' for Portuguese).

    Returns:
        str: The translated text.
    """
    try:
        translator = Translator()
        # Como esperar tradução sem ser assíncrono?
        translated_obj = translator.translate(text_to_translate, dest=dest_lang)
        return translated_obj.text
    except Exception as e:
        st.error(f"Erro ao traduzir: {e}")
        return f"Erro na tradução: {text_to_translate}" # Return original text or an error message


def mostrar_fundamentos(fundamentos: pd.DataFrame):
    """Mostra os fundamentos da ação no Streamlit.
    Args:
        fundamentos (pd.DataFrame): DataFrame contendo os fundamentos da ação.
    """
    st.header(f"{fundamentos['shortName'].values[0]} - {fundamentos['symbol'].values[0]}")
    st.write(f"Nome completo: {fundamentos['longName'].values[0]}")
    if fundamentos['setor_pt'].values[0] == None:
        st.write(f"Setor: {fundamentos['sector'].values[0]}")
    else:
        st.write(f"Setor: {fundamentos['setor_pt'].values[0]}")
    if fundamentos['industria_pt'].values[0] == None:
        st.write(f"Indústria: {fundamentos['industry'].values[0]}")
    else:
        st.write(f"Indústria: {fundamentos['industria_pt'].values[0]}")
    if fundamentos['longBusinessSummary'].values[0] != None:
        #mostrar tradução da descrição do negócio
        st.subheader("Descrição do Negócio")
        # Tenta traduzir a descrição do negócio para português
        descricao = fundamentos['longBusinessSummary'].values[0]
        descricao_traduzida = translate_text(descricao, dest_lang='pt')
        st.write(f'{descricao_traduzida}')
    st.subheader("Fundamentos")
    if fundamentos.empty:
        st.error("Nenhum dado fundamental disponível para o ticker selecionado.")
        return
    # Exibe kpis dos fundamentos no Streamlit
    col1, col2, col3 = st.columns(3)
    with col1:
        if fundamentos['previousClose'].values[0] is not None:
            variacao = fundamentos['regularMarketChangePercent'].values[0]
            st.metric(
                "Último Fechamento",
                f"R$ {fundamentos['previousClose'].values[0]:.2f}",
                delta=f"{variacao:.2f}%",
                delta_color="normal"
            )

        if fundamentos['profitMargins'].values[0] is not None:
            st.metric("Margem de Lucro", f"{fundamentos['profitMargins'].values[0]:.2%}")
        
        if fundamentos['numberOfAnalystOpinions'].values[0] is not None:
            st.metric('Nº de Opiniões de Analistas', fundamentos['numberOfAnalystOpinions'].values[0])

    with col2:
        dy = fundamentos['dividendYield'].values[0]
        if pd.notnull(dy) and dy != 'N/A':
            st.metric("Dividend Yield", f"{float(dy)/100:.2%}")

        
        if fundamentos['recommendationKey'].values[0] is not None:
            if 'buy' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Comprar", delta_color="normal")
                #st.metric("Recomendação", ":cow: [Comprar]", delta_color="normal")
            elif 'sell' in fundamentos['recommendationKey'].values[0]:
                st.metric("Recomendação", "Vender", delta_color="inverse")
                #st.metric("Recomendação", ":bear: [Vender]", delta_color="inverse")
        
        if fundamentos['targetMeanPrice'].values[0] is not None:
            st.metric('Média de Preço alvo', f"R$ {fundamentos['targetMeanPrice'].values[0]:.2f}")

        if fundamentos['targetMedianPrice'].values[0] is not None:
            st.metric('Mediana de Preço alvo', f"R$ {fundamentos['targetMedianPrice'].values[0]:.2f}")

    with col3:
        if fundamentos['lastDividendValue'].values[0] is not None:
            st.metric("Último Dividendo", f"R$ {fundamentos['lastDividendValue'].values[0]:.2f}")
        
        last_div_date = fundamentos['lastDividendDate'].values[0]
        if pd.notnull(last_div_date):
            try:
                data_formatada = pd.to_datetime(last_div_date, unit='s').strftime('%d/%m/%Y')
                st.metric("Data 'ex' do Último Dividendo", data_formatada)
            except Exception as e:
                st.error(f"Erro ao exibir a data do último dividendo: {e}")
        else:
            st.metric("Data do Último Dividendo", "N/A")

        if fundamentos['targetHighPrice'].values[0] is not None:
            st.metric('Preço alvo máximo', f"R$ {fundamentos['targetHighPrice'].values[0]:.2f}")
        
        if fundamentos['targetLowPrice'].values[0] is not None:
            st.metric('Preço alvo mínimo', f"R$ {fundamentos['targetLowPrice'].values[0]:.2f}")
        

    mostrar_fundamentos = st.checkbox('Mostrar todos fundamentos')
    if mostrar_fundamentos:
        for col in fundamentos.columns:
            st.write(f"{col}: {fundamentos[col].values[0]}")