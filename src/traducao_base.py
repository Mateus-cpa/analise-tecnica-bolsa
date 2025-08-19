import json
import os
import time

from googletrans import Translator
import pandas as pd
import streamlit as st


def traduzir_base(colunas=['setor', 'industria']):
    translator = Translator()
    df = pd.read_csv('raw_data/lista_setores.csv', na_values=['N/A'])

    print('Criando pasta bronze_data se não existir...')
    os.makedirs('bronze_data', exist_ok=True)
    print('Pasta bronze_data pronta.')
    
    for coluna in colunas:
        #substituir print por st.
        st.write(f'Processando coluna: {coluna}')
        itens_unicos = set(df[coluna].dropna().unique())
        st.write(f'Itens únicos encontrados: {len(itens_unicos)}')
        itens_unicos = {i for i in itens_unicos if isinstance(i, str) and i.strip() != '' and i != 'N/A' and i != None}
        st.write(f'Itens únicos válidos: {len(itens_unicos)}')
        itens_trad = {}
        for i in itens_unicos:
            tentativas = 0
            traduzido = i
            while tentativas < 3 and traduzido == i:
                try:
                    resultado = translator.translate(i, src='en', dest='pt')
                    traduzido = resultado.text if resultado and hasattr(resultado, 'text') and resultado.text else i
                    tentativas += 1
                    time.sleep(0.5)
                except Exception as e:
                    st.write(f'Erro ao traduzir \"{i}\": {e}')
                    tentativas += 1
            itens_trad[i] = traduzido
            st.write(f'Traduzido: {i} -> {traduzido}')
        json_path = f'bronze_data/traducao_{coluna}.json'
        st.write(f'Salvando traduções em {json_path} ...')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(itens_trad, f, ensure_ascii=False, indent=2)
        st.write(f'Traduções salvas em {json_path}.')
        df[f'{coluna}_pt'] = df[coluna].map(itens_trad)
    csv_path = 'bronze_data/lista_setores_traduzido.csv'
    st.write(f'Salvando DataFrame traduzido em {csv_path} ...')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    st.write(f'DataFrame salvo em {csv_path}.')

if __name__ == "__main__":
    traduzir_base()
    st.write('Tradução concluída.')
