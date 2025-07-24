import json
import os
import time

from googletrans import Translator
import pandas as pd
import streamlit as st


def traduzir_base(colunas=['setor', 'industria']):
    translator = Translator()
    df = pd.read_csv('raw_data/lista_setores.csv', na_values=['N/A'])

    os.makedirs('bronze_data', exist_ok=True)
    
    for coluna in colunas:
        itens_unicos = set(df[coluna].dropna().unique())
        itens_unicos = {i for i in itens_unicos if isinstance(i, str) and i.strip() != '' and i != 'N/A'}
        itens_trad = {}
        for i in itens_unicos:
            try:
                resultado = translator.translate(i, src='en', dest='pt')
                traduzido = resultado.text if resultado and hasattr(resultado, 'text') else i
                itens_trad[i] = traduzido
                time.sleep(0.5)
            except Exception as e:
                print(f'Erro ao traduzir "{i}": {e}')
                itens_trad[i] = i
        with open(f'bronze_data/traducao_{coluna}.json', 'w', encoding='utf-8') as f:
            json.dump(itens_trad, f, ensure_ascii=False, indent=2)
        df[f'{coluna}_pt'] = df[coluna].map(itens_trad)
        print(df[[coluna,f'{coluna}_pt']].head())
    df.to_csv('raw_data/lista_setores_traduzido.csv', index=False, encoding='utf-8-sig')
