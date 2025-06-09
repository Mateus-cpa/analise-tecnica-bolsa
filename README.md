# Estudos de Ciência de Dados para Análise Técnica de Ativos da Bolsa

Este projeto tem como objetivo explorar técnicas de ciência de dados e machine learning aplicadas à análise técnica de ativos negociados na bolsa de valores. Utiliza bibliotecas populares para importar, processar, visualizar e prever cotações de ativos financeiros.

## Objetivos

- Importar dados históricos de ativos financeiros.
- Visualizar séries temporais e gráficos de candlestick.
- Aplicar modelos de machine learning para prever cotações futuras.
- Comparar previsões com valores reais para avaliar desempenho.

## Entrada de Dados

Os dados históricos dos ativos são importados diretamente do Yahoo Finance e processados em formato de tabela utilizando bibliotecas como `pandas` e `yfinance`.

## Visualização dos Dados

Os dados são plotados em gráficos de candlestick para facilitar a análise visual dos movimentos de preço. Para isso, são utilizadas bibliotecas como `matplotlib` e `mplfinance`.

## Previsão e Avaliação

O projeto tenta prever a cotação futura baseada em dados passados, utilizando modelos de machine learning como regressão linear, redes neurais ou outros algoritmos disponíveis em `scikit-learn`. As previsões são comparadas com as cotações reais para avaliar a acurácia dos modelos.

## Estrutura do Projeto


Workspace
(rerun without)
Collecting workspace information
```
├── Análise_técnica_de_ações.ipynb # Notebook principal com análises e experimentos 
├── pyproject.toml # Configuração do Poetry e dependências 
├── poetry.lock # Lockfile de dependências 
├── .python-version # Versão do Python utilizada 
└── README.md # Este arquivo
```

## Instalação de Dependências

Clone o repositório e configure o ambiente virtual com as dependências necessárias:

```bash
git clone git@github.com:Mateus-cpa/analise-tecnica-bolsa.git
cd analise-tecnica-bolsa
python -m venv .venv
source .venv/bin/activate  # No Windows use: .venv\Scripts\activate
pyenv local 3.13.0  
poetry init       # Certifique-se de ter o pyenv e Python 3.13.0 instalados
poetry install
```


## Principais Bibliotecas Utilizadas
- pandas
- yfinance
- matplotlib
- mplfinance
- scikit-learn

### Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias, correções ou novas análises.

## Licença
Este projeto está sob a licença MIT.