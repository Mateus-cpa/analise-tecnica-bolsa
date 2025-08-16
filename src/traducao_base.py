import json
import os
import time

from googletrans import Translator
import pandas as pd


def traduzir_base(colunas=['setor', 'industria']):
    translator = Translator()
    df = pd.read_csv('raw_data/lista_setores.csv', na_values=['N/A'])

    print('Criando pasta bronze_data se não existir...')
    os.makedirs('bronze_data', exist_ok=True)
    print('Pasta bronze_data pronta.')
    
    for coluna in colunas:
        print(f'Processando coluna: {coluna}')
        itens_unicos = set(df[coluna].dropna().unique())
        print(f'Itens únicos encontrados: {len(itens_unicos)}')
        itens_unicos = {i for i in itens_unicos if isinstance(i, str) and i.strip() != '' and i != 'N/A'}
        print(f'Itens únicos válidos: {len(itens_unicos)}')
        itens_trad = {}
        for i in itens_unicos:
            traduzido = i
            while i == traduzido:  # Garante que a tradução seja diferente
                try:
                    resultado = translator.translate(i, src='en', dest='pt')
                    traduzido = resultado.text if resultado and hasattr(resultado, 'text') else i
                    itens_trad[i] = traduzido
                    print(f'Traduzido: {i} -> {traduzido}')
                    time.sleep(0.5)
                except Exception as e:
                    print(f'Erro ao traduzir "{i}": {e}')
                    itens_trad[i] = i
        json_path = f'bronze_data/traducao_{coluna}.json'
        print(f'Salvando traduções em {json_path} ...')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(itens_trad, f, ensure_ascii=False, indent=2)
        print(f'Traduções salvas em {json_path}.')
        df[f'{coluna}_pt'] = df[coluna].map(itens_trad)
    csv_path = 'bronze_data/lista_setores_traduzido.csv'
    print(f'Salvando DataFrame traduzido em {csv_path} ...')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'DataFrame salvo em {csv_path}.')
