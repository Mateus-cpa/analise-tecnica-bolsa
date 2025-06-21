import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt


import streamlit as st
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler # para normalizar os dados
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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

    """#normalizando novamente os dados para LSTM
    scaler = StandardScaler()
    df_para_lstm = pd.DataFrame(acao['Close'])
    st.subheader('Testes previsão')
    df_scaled = scaler.fit_transform(df_para_lstm)
    st.write(f'tamanho total {df_scaled.shape}')

    #separando as linhas para o teste
    train = df_scaled[:qtd_linhas_treino]
    test = df_scaled[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]
    st.write(f'tam treino {train.shape}, tam teste {test.shape}')

    #converter uma array na matriz dataframe
    def create_df(df, steps=1):
        dataX, dataY = [], []
        for i in range(len(df)-steps-1):
            a = df[i:(i+steps), 0]
            dataX.append(a)
            dataY.append(df[i + steps, 0])
        return np.array(dataX), np.array(dataY)

    #gerando dados de treino e teste
    steps = 15
    X_train, y_train = create_df(train, steps)
    X_test, y_test = create_df(test, steps)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    #gerando os dados que o modelo espera
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) #recriando o dados de treinamento. 1 para a quantidade de features recebidas
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #montando a rede
    model = Sequential()
    model.add(LSTM(35,
                return_sequences=True, #return_sequences input para o próximo teste
                input_shape=(steps, 1)))
    model.add(LSTM(35,
                return_sequences=True)) #segundo teste, envia para o próximo teste
    model.add(LSTM(35)) #teste final
    model.add(Dropout(0.2)) # regulariza a informação para não causar overfitting
    model.add(Dense(1)) # saída de uma 1 neurônio
    model.compile(loss='mse',
                optimizer='adam') #um dos mais utilizados
    model.summary()


    # treinar modelo
    validation = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=15, #quantidade de informação por vez
                        validation_data=(X_test, y_test),
                        verbose=2) #ver como as informações estão sendo lotadas
    
    fig, ax = plt.subplots()
    ax.plot(validation.history['loss'], label='Training loss')
    ax.plot(validation.history['val_loss'], label='Validation loss')
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Histórico de Loss do Modelo")
    plt.close(fig)

    # visualizando a previsão
    prev = model.predict(X_test)
    prev = scaler.inverse_transform(prev) # dados desnormalizados
    df_lstm = df_para_lstm.iloc[qtd_linhas_treino:qtd_linhas_treino,:]
    st.write(df_lstm.head())
    df_lstm['prev_lstm'] = prev
    plt.plot(df_lstm, x= df_lstm.index, y = df_lstm.prec_lstm, label='Previsão')
    plt.plot(df_lstm, x= df_lstm.index, y = df_lstm.Close, label='Real')
    plt.legend()
    plt.show()

    #testar modelo preditivo par apróximos 10 dias
    length_test = len(test) #tamanho predefinido de teste
    print(length_test)

    #pegar os últimos doas que são o tamano dos steps
    days_input_steps = length_test - steps

    print(f' Início teste: {df_para_lstm.iloc[qtd_linhas_treino + days_input_steps]}')


    #transforma em array
    input_steps = test[days_input_steps:]
    input_steps = np.array(input_steps).reshape(1,-1) #ficar com 1 linha -1

    #transformar em lista
    list_output_steps = list(input_steps)
    list_output_steps = list_output_steps[0].tolist() #transforma em lista

    print("dados de entrada:")
    for i in range(0,len(list_output_steps)):
        print(f'{i} => {list_output_steps[i]}')
    
    #loop para prever os próximos 10 dias
    pred_output = [] # cria uma lista para recebre os 10dias futuros
    i=0
    n_futuro = 15 # qtde de 15 dias

    while i < n_futuro:

    if (len(list_output_steps) > steps):

        input_steps = np.array(list_output_steps[1:])
        print(f'{i}º dia: valores de entrada => {input_steps}')
        input_steps = input_steps.reshape(1,-1)
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model.predict(input_steps, verbose=0)
        print(f'{i}º dia: previsão => {pred}')
        list_output_steps.extend(pred[0].tolist())
        list_output_steps = list_output_steps[1:]
        pred_output.extend(pred[0].tolist())
        i+=1
    else:
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model.predict(input_steps, verbose=0) #previsão
        print(f'{i}º dia: previsão => {pred[0]}')
        list_output_steps.extend(pred[0].tolist()) #deslocamento do tamanho da lista
        print(f'Tamanho da lista {len(list_output_steps)}')
        pred_output.extend(pred[0].tolist())
        i+=1

    print(pred_output)

    # desnormaliza a saída
    prev = scaler.inverse_transform(pd.DataFrame(pred_output))
    prev = np.array(prev).reshape(1,-1)
    prev = list(prev)
    previsao_10_dias = prev[0].tolist()
    for i in range(0,len(previsao_10_dias)):
        print(f'{[i]} => {previsao_10_dias[i]}')

    #pegar as datas da previsão
    datas_prev = acao_prev.index
    predict_dates = pd.date_range(datas_prev[-1] + pd.DateOffset(1), periods=n_futuro,freq='b').tolist()

    # cria dataframe de previsão
    forecast_dates = []
    for i in predict_dates:
    forecast_dates.append(i.date())

    df_forecast = pd.DataFrame({'data_pregao': np.array(forecast_dates), 'preco_previsao_fechamento': previsao_10_dias})
    df_forecast['data_pregao'] = pd.to_datetime(df_forecast['data_pregao'])
    df_forecast.set_index('data_pregao', inplace=True)

    #concatenar datagframes df e df_forecast
    df = pd.concat([acao_final,df_forecast])
    df.tail(20)

    grafico_forecast = go.Figure()
    grafico_forecast.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Real'))
    grafico_forecast.add_trace(go.Scatter(x=df.index, y=df['preco_previsao_fechamento'], name='Previsão'))
    grafico_forecast.show()"""

    return acao_concat

if __name__ == "__main__":
    acao_com_preditivo()
    