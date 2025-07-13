import json
import os
import time

from googletrans import Translator
import pandas as pd
import streamlit as st


def traduzir_base(colunas=['setor', 'industria']):
    translator = Translator()
    
    raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bronze_data'))
    
    os.makedirs(bronze_data_dir, exist_ok=True)
    
    setores_path = os.path.join(raw_data_dir, 'lista_setores.csv')
    df = pd.read_csv(setores_path)

    for coluna in colunas:
        itens_unicos = set(df[coluna].dropna().unique())
        itens_unicos = {i for i in itens_unicos if isinstance(i, str) and i.strip() != ''}
        
        # Carrega traduções existentes para evitar retradução
        traducao_file = os.path.join(bronze_data_dir, f'traducao_{coluna}.json')
        if os.path.exists(traducao_file):
            with open(traducao_file, 'r', encoding='utf-8') as f:
                itens_trad = json.load(f)
        else:
            itens_trad = {}

        # Filtra itens que já foram traduzidos
        itens_a_traduzir = [item for item in itens_unicos if item not in itens_trad]
        
        if not itens_a_traduzir:
            st.info(f"Todas as entradas para '{coluna}' já estão traduzidas. Pulando tradução.")
        else:
            st.info(f"Traduzindo {len(itens_a_traduzir)} itens únicos para '{coluna}'...")
            progress_bar_traducao = st.progress(0)
            status_text_traducao = st.empty()

            for idx, i in enumerate(itens_a_traduzir):
                retries = 3 # Número de tentativas para cada tradução
                for attempt in range(retries):
                    try:
                        status_text_traducao.text(f"Traduzindo '{i}' ({idx + 1}/{len(itens_a_traduzir)})")
                        resultado = translator.translate(i, src='en', dest='pt')
                        traduzido = resultado.text if resultado and hasattr(resultado, 'text') else i
                        itens_trad[i] = traduzido
                        break # Sai do loop de tentativas se a tradução for bem-sucedida
                    except Exception as e:
                        print(f'Erro ao traduzir "{i}" (tentativa {attempt + 1}/{retries}): {e}')
                        if attempt < retries - 1:
                            wait_time = 2 ** attempt # Backoff exponencial (1, 2, 4 segundos)
                            time.sleep(wait_time)
                        else:
                            itens_trad[i] = i # fallback: mantém o original após todas as tentativas

                progress_bar_traducao.progress((idx + 1) / len(itens_a_traduzir))

            # Salva o dicionário de tradução individual (atualizado com novas traduções)
            with open(traducao_file, 'w', encoding='utf-8') as f:
                json.dump(itens_trad, f, ensure_ascii=False, indent=2)
            
        # Adiciona coluna traduzida ao DataFrame
        df[f'{coluna}_pt'] = df[coluna].map(itens_trad)
        print(df[[coluna,f'{coluna}_pt']].head())

    # Salva o DataFrame atualizado em um novo arquivo (mantendo o original intocado se desejar)
    df.to_csv(os.path.join(raw_data_dir, 'lista_setores_traduzido.csv'), index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    traduzir_base()