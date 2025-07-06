import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import json

# Bibliotecas importadas
import streamlit as st

# MLs
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso

@st.cache_data
def selecionar_colunas(features, labels):
    features_list = ('Open','High','Low','Volume','MM5', 'MM21','MM72','MM200')
    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features, labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))
    k_best_features_final = dict(ordered_pairs[:15])
    k_best_features_filtrado = {k: v for k, v in k_best_features_final.items() if v > 50}
    melhores_colunas = list(k_best_features_filtrado.keys())
    return melhores_colunas, k_best_features_filtrado

@st.cache_resource
def treinar_lr(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    return lr

@st.cache_resource
def treinar_rn(X_train, y_train):
    rn = MLPRegressor(max_iter=2000)
    rn.fit(X_train, y_train)
    return rn

@st.cache_resource
def treinar_rn_hp(X_train, y_train):
    rn_hp = MLPRegressor()
    parameter_space = {
        'hidden_layer_sizes': [(i,) for i in list(range(1, 21))],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    search = GridSearchCV(rn_hp, parameter_space, n_jobs=-1, cv=5)
    search.fit(X_train, y_train)
    return search.best_estimator_

@st.cache_resource
def treinar_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf

@st.cache_resource
def treinar_gb(X_train, y_train):
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    return gb

@st.cache_resource
def treinar_svr(X_train, y_train):
    svr = SVR()
    svr.fit(X_train, y_train)
    return svr

@st.cache_resource
def treinar_ridge(X_train, y_train):
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    return ridge

@st.cache_resource
def treinar_lasso(X_train, y_train):
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    return lasso

@st.cache_data
def normalizar_features(features):
    scaler = MinMaxScaler().fit(features)
    features_scale = scaler.transform(features)
    return scaler, features_scale

def prever_n_dias(acao, modelos, scaler, melhores_colunas, n=10):
    acao_prev = acao.copy()
    previsoes_futuras = []
    for i in range(n):
        proxima_data = acao_prev.index[-1] + BDay(1)
        nova_linha = {}
        temp = acao_prev[-200:].copy()
        nova_linha['Open'] = temp['Close'].iloc[-1]
        nova_linha['High'] = temp['Close'].iloc[-1]
        nova_linha['Low'] = temp['Close'].iloc[-1]
        nova_linha['Volume'] = temp['Volume'].iloc[-1]
        for mm in [5, 21, 72, 200]:
            nova_linha[f'MM{mm}'] = pd.concat([temp['Close'], pd.Series([temp['Close'].iloc[-1]])]).rolling(window=mm).mean().iloc[-1]
        features_novo_dia = pd.DataFrame([nova_linha], index=[proxima_data])
        features_novo_dia = features_novo_dia[melhores_colunas]
        features_novo_dia_scaled = scaler.transform(features_novo_dia)
        for nome, modelo in modelos.items():
            pred = modelo.predict(features_novo_dia_scaled)[0]
            nova_linha[f'previsao_{nome}'] = pred
        nova_linha['Close'] = nova_linha.get('previsao_regressao_linear', np.nan)
        if proxima_data in acao_prev.index:
            acao_prev.loc[proxima_data] = nova_linha
        else:
            acao_prev = pd.concat([acao_prev, pd.DataFrame([nova_linha], index=[proxima_data])])
        previsoes_futuras.append(nova_linha)
    return acao_prev

def acao_com_preditivo(acao):
    acao_prev = acao.copy()
    acao_prev = acao_prev.drop(columns=['mudanca_tendencia', 'marcador'])
    acao_prev = acao_prev.dropna()
    qtd_linhas = len(acao_prev)
    qtd_linhas_treino = round(.70 * qtd_linhas)
    qtd_linhas_teste = qtd_linhas - qtd_linhas_treino
    features = acao_prev.drop(columns=['Close'])
    labels = acao_prev['Close']

    melhores_colunas, k_best_features_filtrado = selecionar_colunas(features, labels)
    
    features = acao_prev.loc[:, melhores_colunas]
    scaler, features_scale = normalizar_features(features)

    X_train = features_scale[:qtd_linhas_treino]
    X_test = features_scale[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]
    y_train = labels[:qtd_linhas_treino]
    y_test = labels[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]
    
    modelos = {}
    st.subheader('Modelos de Machine Learning')
    if st.checkbox(f"Dados de treino de ML das {len(melhores_colunas)} colunas de dataframe com k-means acima de 50:"):
        st.write(k_best_features_filtrado)
        st.write(f'Divisão das linhas /// FEATURES: {features_scale.shape}, TREINO: {len(X_train)}, TESTE: {len(X_test)}')

    col1, col2, col3, col4 = st.columns(4)
    coeficientes_modelos = {}
    #atribuir valor 0 a todos modelos
    for modelo in ['regressao_linear', 'rede_neural', 'hiper_parametro', 'random_forest', 'gradient_boosting', 'svr', 'ridge', 'lasso']:
        coeficientes_modelos[modelo] = 0.0
    

    if col1.checkbox('Regressão linear', value=True):
        lr = treinar_lr(X_train, y_train)
        pred_lr = lr.predict(X_test)
        cd_lr = r2_score(y_test, pred_lr)
        col1.write(f'Coef: {cd_lr * 100:.2f}')
        coeficientes_modelos['regressao_linear'] = cd_lr * 100
        modelos['regressao_linear'] = lr

    if col1.checkbox('Rede neural', value=True):
        rn = treinar_rn(X_train, y_train)
        pred_rn = rn.predict(X_test)
        cd_rn = rn.score(X_test, y_test)
        col1.write(f'Coef: {cd_rn * 100:.2f}')
        coeficientes_modelos['rede_neural'] = cd_rn * 100
        modelos['rede_neural'] = rn

    if col2.checkbox('Hyperparametros (pesado!)'):
        clf = treinar_rn_hp(X_train, y_train)
        pred_rnhp = clf.predict(X_test)
        cd_rnhp = clf.score(X_test, y_test)
        col2.write(f'Coef: {cd_rnhp * 100:.2f}')
        coeficientes_modelos['hiper_parametro'] = cd_rnhp * 100
        modelos['rede_neural_hiper_parameter'] = clf

    if col2.checkbox('Random Forest Regression', value=True):
        rf = treinar_rf(X_train, y_train)
        pred_rf = rf.predict(X_test)
        cd_rf = rf.score(X_test, y_test)
        col2.write(f'Coef: {cd_rf * 100:.2f}')
        coeficientes_modelos['random_forest'] = cd_rf * 100
        modelos['random_forest'] = rf

    if col3.checkbox('Gradient Boosting Regr.', value=True):
        gb = treinar_gb(X_train, y_train)
        pred_gb = gb.predict(X_test)
        cd_gb = gb.score(X_test, y_test)
        col3.write(f'Coef: {cd_gb * 100:.2f}')
        coeficientes_modelos['gradient_boosting'] = cd_gb * 100
        modelos['gradient_boosting'] = gb

    if col3.checkbox('Support Vector Regression', value=True):
        svr = treinar_svr(X_train, y_train)
        pred_svr = svr.predict(X_test)
        cd_svr = svr.score(X_test, y_test)
        col3.write(f'Coef: {cd_svr * 100:.2f}')
        coeficientes_modelos['svr'] = cd_svr * 100
        modelos['svr'] = svr

    if col4.checkbox('Regressão Ridge', value=True):
        ridge = treinar_ridge(X_train, y_train)
        cd_ridge = ridge.score(X_test, y_test)
        col4.write(f'Coef: {cd_ridge * 100:.2f}')
        coeficientes_modelos['ridge'] = cd_ridge * 100
        modelos['ridge'] = ridge

    if col4.checkbox('Regressão Lasso', value=True):
        lasso = treinar_lasso(X_train, y_train)
        cd_lasso = lasso.score(X_test, y_test)
        col4.write(f'Coef: {cd_lasso * 100:.2f}')
        coeficientes_modelos['lasso'] = cd_lasso * 100
        modelos['lasso'] = lasso

    # Salva coeficientes dos modelos em json
    with open('bronze_data/coeficientes_modelos.json', 'w') as f:
        json.dump(coeficientes_modelos, f)

    previsao_supervisionada = features_scale[qtd_linhas_teste:qtd_linhas]
    data_pregao_full = acao_prev.index
    data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]
    res_full = acao_prev['Close']
    res = res_full[qtd_linhas_teste:qtd_linhas]

    # Previsões de todos os modelos
    pred_lr = modelos['regressao_linear'].predict(previsao_supervisionada) if 'regressao_linear' in modelos else None
    pred_rn = modelos['rede_neural'].predict(previsao_supervisionada) if 'rede_neural' in modelos else None
    pred_rnhp = modelos['rede_neural_hiper_parameter'].predict(previsao_supervisionada) if 'rede_neural_hiper_parameter' in modelos else None
    pred_rf = modelos['random_forest'].predict(previsao_supervisionada) if 'random_forest' in modelos else None
    pred_gb = modelos['gradient_boosting'].predict(previsao_supervisionada) if 'gradient_boosting' in modelos else None
    pred_svr = modelos['svr'].predict(previsao_supervisionada) if 'svr' in modelos else None
    pred_ridge = modelos['ridge'].predict(previsao_supervisionada) if 'ridge' in modelos else None
    pred_lasso = modelos['lasso'].predict(previsao_supervisionada) if 'lasso' in modelos else None
  
    df = pd.DataFrame({'data_pregao': data_pregao, 'real': res})
    if pred_lr is not None:
        df['previsao_regressao_linear'] = pred_lr
    if pred_rn is not None:
        df['previsao_rede_neural'] = pred_rn
    if pred_rnhp is not None:
        df['previsao_rede_neural_hiper_parameter'] = pred_rnhp
    if pred_rf is not None:
        df['previsao_random_forest'] = pred_rf
    if pred_gb is not None:
        df['previsao_gradient_boosting'] = pred_gb
    if pred_svr is not None:
        df['previsao_svr'] = pred_svr
    if pred_ridge is not None:
        df['previsao_ridge'] = pred_ridge
    if pred_lasso is not None:
        df['previsao_lasso'] = pred_lasso

    nova_linha = {'real': None}
    if pred_lr is not None:
        nova_linha['previsao_regressao_linear'] = modelos['regressao_linear'].predict([features_scale[-1]])[0]
    if pred_rn is not None:
        nova_linha['previsao_rede_neural'] = modelos['rede_neural'].predict([features_scale[-1]])[0]
    if pred_rnhp is not None:
        nova_linha['previsao_rede_neural_hiper_parameter'] = modelos['rede_neural_hiper_parameter'].predict([features_scale[-1]])[0]
    if pred_rf is not None:
        nova_linha['previsao_random_forest'] = modelos['random_forest'].predict([features_scale[-1]])[0]
    if pred_gb is not None:
        nova_linha['previsao_gradient_boosting'] = modelos['gradient_boosting'].predict([features_scale[-1]])[0]
    if pred_svr is not None:
        nova_linha['previsao_svr'] = modelos['svr'].predict([features_scale[-1]])[0]
    if pred_ridge is not None:
        nova_linha['previsao_ridge'] = modelos['ridge'].predict([features_scale[-1]])[0]
    if pred_lasso is not None:
        nova_linha['previsao_lasso'] = modelos['lasso'].predict([features_scale[-1]])[0]

    previsoes_cols = [col for col in df.columns if col.startswith('previsao_')]
    previsoes = df[previsoes_cols]
    acao_concat = pd.concat([acao, previsoes], axis=1)

    #previsão dos próximos 10 dias
    df_prev_10_dias = prever_n_dias(acao_prev, modelos, scaler, melhores_colunas, n=10)
    df_prev_10_dias.index.name = acao_concat.index.name

    for col in acao_concat.columns:
        if col not in df_prev_10_dias.columns:
            df_prev_10_dias[col] = np.nan
    for col in df_prev_10_dias.columns:
        if col not in acao_concat.columns:
            acao_concat[col] = np.nan

    df_prev_10_dias = df_prev_10_dias[acao_concat.columns]
    acao_concat = pd.concat([acao_concat, df_prev_10_dias], axis=0)
    acao_concat = acao_concat[~acao_concat.index.duplicated(keep='first')]

    # Apaga valores de 'Close' nos dias de previsão futura (após o último índice real)
    ultimo_indice_real = acao.index[-1]
    acao_concat.loc[acao_concat.index > ultimo_indice_real, 'Close'] = np.nan

    return acao_concat

if __name__ == "__main__":
    acao_com_preditivo()
    # Fonte:
    # - https://www.infomoney.com.br/guias/analise-tecnica/
    # - Video previsão 10 dias: https://www.youtube.com/watch?v=CvfAx3_nGME&list=PL1woXE9p74ASlH4i2QQytmASjle8Bt8An&index=7