import pandas as pd

import streamlit as st
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler # para normalizar os dados
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM, Dropout

def acao_com_preditivo(acao):
    #ajustar valor fechamento para o dia anterior
    acao_prev = acao.copy()
    acao_prev['Close'] = acao_prev['Close'].shift(-1) # passa o valor de fechamento para o dia anterior

    #retirando dados vazios
    acao_prev = acao_prev.drop(columns = ['mudanca_tendencia', 'marcador'])
    acao_prev = acao_prev.dropna()

    #verificando quantidade de linhas
    qtd_linhas = len(acao_prev)

    qtd_linhas_treino= round(.70 * qtd_linhas)
    qtd_linhas_teste= qtd_linhas - qtd_linhas_treino
    qtd_linhas_validacao = qtd_linhas -1

    st.write(f"linhas treino = 0:{qtd_linhas_treino} \n")
    st.write(f"linhas teste = {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste -1} \n")
    st.write(f"linhas validação = {qtd_linhas_validacao}")
    
    #separando as features e labels
    features = acao_prev.drop(columns = ['Close']) # isola o dataframe da coluna de fechamento
    labels = acao_prev['Close'] # seleciona apenas a coluna de fechamento para previsão

    #testando quais features têm melhores resultados de previsão
    features_list = ('Open','High','Low','Volume','MM5', 'MM21','MM72','MM200')#, 'rsi')

    k_best_features = SelectKBest(k='all') # seleciona as melhores features (colunas)
    k_best_features.fit_transform(features, labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores) # transforma em dicionário os nomes das features e a pontuação de afinidade com a coluna de avaliação
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1]))) # ordena os maiores aos menores

    k_best_features_final = dict(ordered_pairs[:15])
    best_features = k_best_features_final.keys()

    st.write("Melhores features:")
    st.write(k_best_features_final)

    columns_features = []
    for k,v in k_best_features_final.items():
        if v > 50:
            columns_features.append(k)
    st.write(f'novas colunas de dataframe com k-means acima de 50: {columns_features}')

    #separando as features escolhidas
    features = acao_prev.loc[:,columns_features]

    #Normalizando os dados de entrada(features)

    # Gerando o novo padrão
    scaler = MinMaxScaler().fit(features)
    features_scale = scaler.transform(features)
    st.write(f'features: {features_scale.shape}')
    
    #Separa os dados de treino teste e validação
    X_train = features_scale[:qtd_linhas_treino]
    X_test = features_scale[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]

    y_train = labels[:qtd_linhas_treino]
    y_test = labels[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]

    st.write(f'x treino: {len(X_train)}, y treino: {len(y_train)}')
    st.write(f'x teste: {len(X_test)}, y teste: {len(y_test)}')

    st.subheader('Coeficientes')
    col1, col2, col3 = st.columns(3)
    if col1.checkbox('Regressão linear', value=True):
        #treinamento usando regressão linear
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        pred= lr.predict(X_test)
        cd_lr =r2_score(y_test, pred)
        st.write(f'Coeficiente de determinação regressão linear:{cd_lr * 100:.2f}')

    if col2.checkbox('Rede neural', value=True):
        #rede neural
        rn = MLPRegressor(max_iter=2000)
        rn.fit(X_train, y_train)
        pred= rn.predict(X_test)
        cd_rn = rn.score(X_test, y_test)
        st.write(f'Coeficiente de determinação rede neural:{cd_rn * 100:.2f}')

    if col3.checkbox('Hyperparametros'):
        #rede neural com ajuste hyper parameters
        rn_hp = MLPRegressor()
        parameter_space = {
                'hidden_layer_sizes': [(i,) for i in list(range(1, 21))], # camada escondida, quantos neurônios. De 1 a 21, neste caso
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
        search = GridSearchCV(rn_hp, parameter_space, n_jobs=-1, cv=5)
        search.fit(X_train,y_train)
        clf = search.best_estimator_
        pred= search.predict(X_test)
        cd_rnhp = search.score(X_test, y_test)
        st.write(f'Coeficiente de determinação rede neural hiper parameters: {cd_rnhp * 100:.2f}')
    
    #dataframe com a previsão
    previsao = features_scale[qtd_linhas_teste:qtd_linhas]
    data_pregao_full = acao_prev.index
    data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]
    res_full = acao_prev['Close']
    res = res_full[qtd_linhas_teste:qtd_linhas]
    pred_lr = lr.predict(previsao) #utilizando regressão linear
    pred_rn = rn.predict(previsao) #utilizando rede neural
    try:
        pred_rnhp = clf.predict(previsao) #utilizando rede neural com ajuste hyper parameters
    except:
        pred_rnhp = None
    try:
        df=pd.DataFrame({'data_pregao':data_pregao,
                         'real':res,
                         'previsao_regressao_linear':pred_lr,
                         'previsao_rede_neural': pred_rn,
                         'previsao_rede_neural_hiper_parameter': pred_rnhp}) # diferencial
    except:
        df=pd.DataFrame({'data_pregao':data_pregao,
                         'real':res,
                         'previsao_regressao_linear':pred_lr,
                         'previsao_rede_neural': pred_rn})
    df.real = df.real.shift(+1)
    df.set_index('data_pregao', inplace=True)
    # Concatenar previsões ao DataFrame original acao
    # Alinhar pelo índice (data)
    previsoes = df[['previsao_regressao_linear', 'previsao_rede_neural']]
    if 'previsao_rede_neural_hiper_parameter' in df.columns:
        previsoes = pd.concat([previsoes, df[['previsao_rede_neural_hiper_parameter']]], axis=1)
    acao_concat = pd.concat([acao, previsoes], axis=1)

    return acao_concat

if __name__ == "__main__":
    acao_com_preditivo()
    