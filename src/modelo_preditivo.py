import pandas as pd
from pandas.tseries.offsets import BDay

import streamlit as st

#MLs
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler # para normalizar os dados
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso



def acao_com_preditivo(acao):
    acao_prev = acao.copy()

    # NÃO faça o shift(-1) na coluna Close!
    # acao_prev['Close'] = acao_prev['Close'].shift(-1)

    # Retirando dados vazios
    acao_prev = acao_prev.drop(columns=['mudanca_tendencia', 'marcador'])
    acao_prev = acao_prev.dropna()

    # Verificando quantidade de linhas
    qtd_linhas = len(acao_prev)
    qtd_linhas_treino = round(.70 * qtd_linhas)
    qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

    # Separando as features e labels
    features = acao_prev.drop(columns=['Close'])
    labels = acao_prev['Close']

    # Seleção de features (mantém igual)
    features_list = ('Open','High','Low','Volume','MM5', 'MM21','MM72','MM200')
    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features, labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))
    k_best_features_final = dict(ordered_pairs[:15])
    melhores_colunas = []
    k_best_features_filtrado = {k: v for k, v in k_best_features_final.items() if v > 50}
    for k in k_best_features_filtrado:
        melhores_colunas.append(k)
    st.write(f'encontradas {len(melhores_colunas)} colunas de dataframe com k-means acima de 50:')
    st.write(k_best_features_filtrado)

    # Separando as features escolhidas
    features = acao_prev.loc[:, melhores_colunas]

    # Normalizando os dados de entrada (features)
    scaler = MinMaxScaler().fit(features)
    features_scale = scaler.transform(features)

    # Separa os dados de treino, teste e validação
    X_train = features_scale[:qtd_linhas_treino]
    X_test = features_scale[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]
    y_train = labels[:qtd_linhas_treino]
    y_test = labels[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]

    st.write(f'Divisão das linhas /// FEATURES: {features_scale.shape}, TREINO: {len(X_train)}, TESTE: {len(X_test)}')

    st.subheader('Modelos de Machine Learning')
    col1, col2, col3, col4 = st.columns(4)
    coeficientes_modelos = {}

    if col1.checkbox('Regressão linear', value=True):
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        cd_lr = r2_score(y_test, pred_lr)
        col1.write(f'Coef: {cd_lr * 100:.2f}')
        coeficientes_modelos['Regressão Linear'] = cd_lr * 100

    if col1.checkbox('Rede neural', value=True):
        rn = MLPRegressor(max_iter=2000)
        rn.fit(X_train, y_train)
        pred_rn = rn.predict(X_test)
        cd_rn = rn.score(X_test, y_test)
        col1.write(f'Coef: {cd_rn * 100:.2f}')
        coeficientes_modelos['Rede Neural'] = cd_rn * 100

    if col2.checkbox('Hyperparametros (pesado!)'):
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
        clf = search.best_estimator_
        pred_rnhp = search.predict(X_test)
        cd_rnhp = search.score(X_test, y_test)
        col2.write(f'Coef: {cd_rnhp * 100:.2f}')
        coeficientes_modelos['Rede Neural HP'] = cd_rnhp * 100

    if col2.checkbox('Random Forest Regr.', value=True):
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        cd_rf = rf.score(X_test, y_test)
        col2.write(f'Coef: {cd_rf * 100:.2f}')
        coeficientes_modelos['Random Forest'] = cd_rf * 100

    if col3.checkbox('Gradient Boosting Regr.', value=True):
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        pred_gb = gb.predict(X_test)
        cd_gb = gb.score(X_test, y_test)
        col3.write(f'Coef: {cd_gb * 100:.2f}')
        coeficientes_modelos['Gradient Boosting'] = cd_gb * 100

    if col3.checkbox('Support Vector Regression', value=True):
        svr = SVR()
        svr.fit(X_train, y_train)
        pred_svr = svr.predict(X_test)
        cd_svr = svr.score(X_test, y_test)
        col3.write(f'Coef: {cd_svr * 100:.2f}')
        coeficientes_modelos['SVR'] = cd_svr * 100

    if col4.checkbox('Regressão Ridge', value=True):
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        cd_ridge = ridge.score(X_test, y_test)
        col4.write(f'Coef: {cd_ridge * 100:.2f}', value=True)
        coeficientes_modelos['Ridge'] = cd_ridge * 100

    if col4.checkbox('Regressão Lasso', value=True):
        lasso = Lasso()
        lasso.fit(X_train, y_train)
        cd_lasso = lasso.score(X_test, y_test)
        col4.write(f'Coef: {cd_lasso * 100:.2f}')
        coeficientes_modelos['Lasso'] = cd_lasso * 100

    # Plotando gráfico dos coeficientes
    if coeficientes_modelos:
        coef_filtrados = {k: v for k, v in coeficientes_modelos.items() if v > 0}
        coef_ordenados = dict(sorted(coef_filtrados.items(), key=lambda item: item[1], reverse=True))
        st.write('Comparativo dos Coeficientes de Determinação (%)')
        st.bar_chart(coef_ordenados, horizontal=True)

    # dataframe com a previsão
    previsao_supervivionada = features_scale[qtd_linhas_teste:qtd_linhas]
    data_pregao_full = acao_prev.index
    data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]
    res_full = acao_prev['Close']
    res = res_full[qtd_linhas_teste:qtd_linhas]

    # Previsões de todos os modelos
    pred_lr = lr.predict(previsao_supervivionada) if 'lr' in locals() else None
    pred_rn = rn.predict(previsao_supervivionada) if 'rn' in locals() else None
    try:
        pred_rnhp = clf.predict(previsao_supervivionada)
    except:
        pred_rnhp = None
    pred_rf = rf.predict(previsao_supervivionada) if 'rf' in locals() else None
    pred_gb = gb.predict(previsao_supervivionada) if 'gb' in locals() else None
    pred_svr = svr.predict(previsao_supervivionada) if 'svr' in locals() else None
    pred_ridge = ridge.predict(previsao_supervivionada) if 'ridge' in locals() else None
    pred_lasso = lasso.predict(previsao_supervivionada) if 'lasso' in locals() else None

    # Monta o DataFrame com as previsões
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

    # Cria linha extra para o próximo pregão
    proxima_data = df['data_pregao'].iloc[-1] + BDay(1)
    nova_linha = {'real': None}
    if pred_lr is not None:
        nova_linha['previsao_regressao_linear'] = lr.predict([features_scale[-1]])[0]
    if pred_rn is not None:
        nova_linha['previsao_rede_neural'] = rn.predict([features_scale[-1]])[0]
    if pred_rnhp is not None:
        nova_linha['previsao_rede_neural_hiper_parameter'] = clf.predict([features_scale[-1]])[0]
    if pred_rf is not None:
        nova_linha['previsao_random_forest'] = rf.predict([features_scale[-1]])[0]
    if pred_gb is not None:
        nova_linha['previsao_gradient_boosting'] = gb.predict([features_scale[-1]])[0]
    if pred_svr is not None:
        nova_linha['previsao_svr'] = svr.predict([features_scale[-1]])[0]
    if pred_ridge is not None:
        nova_linha['previsao_ridge'] = ridge.predict([features_scale[-1]])[0]
    if pred_lasso is not None:
        nova_linha['previsao_lasso'] = lasso.predict([features_scale[-1]])[0]

    # Adiciona a nova linha ao DataFrame
    nova_linha_df = pd.DataFrame([nova_linha], index=[proxima_data])
    df.set_index('data_pregao', inplace=True)
    df = pd.concat([df, nova_linha_df])


    # Concatenar previsões ao DataFrame original acao
    previsoes_cols = [col for col in df.columns if col.startswith('previsao_')]
    previsoes = df[previsoes_cols]
    acao_concat = pd.concat([acao, previsoes], axis=1)

    return acao_concat

if __name__ == "__main__":
    acao_com_preditivo()
    """Fonte:
- https://www.infomoney.com.br/guias/analise-tecnica/
- Video previsão 10 dias: # https://www.youtube.com/watch?v=CvfAx3_nGME&list=PL1woXE9p74ASlH4i2QQytmASjle8Bt8An&index=7"""
    