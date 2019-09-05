# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=sorder,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except BaseException:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0]
    D_params = [0]
    Q_params = [0]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def create_dataset(input_file, output_file, look_back_window_size=9, ):
    print("Start create data set")
    df = pd.read_csv(input_file, index_col=0)
    dataX, dataY = [], []

    for i in range(len(df) - look_back_window_size - 1):
        dataX.append(df.iloc[i: i + look_back_window_size, 5])
        dataY.append(df.iloc[i + look_back_window_size, 5] - df.iloc[i + look_back_window_size - 1, 5])
        if i % 100 == 0:
            print("creating data, step: ", i)

    dataY = np.array(dataY)
    dataX = np.array(dataX)
    dataY[dataY >= 0.0001] = 1
    dataY[dataY <= -0.0001] = 2
    dataY[dataY < 0.0001] = 0
    print((dataY == 0).sum())
    print((dataY == 1).sum())
    print((dataY == 2).sum())
    # one-hot encoding it
    dataY = to_categorical(dataY, num_classes=3)

    data = np.hstack((dataX, dataY))

    pd.DataFrame(data).to_csv(output_file)
    print("Create data set done!!!")


if __name__ == '__main__':
    # create_dataset("./data/EURUSD_m15_test.csv", "./data/EURUSD_time_serires_test.csv")
    # create_dataset("./data/EURUSD_m15_train.csv", "./data/EURUSD_time_serires_train.csv")
    train_df = pd.read_csv("./data/EURUSD_time_serires_train.csv", index_col=0)
    test_df = pd.read_csv("./data/EURUSD_time_serires_test.csv", index_col=0)
    train_df = train_df.to_numpy()
    test_df = test_df.to_numpy()
    X_train, Y_train = train_df[:, :-3], train_df[:, -3:]
    X_test, Y_test = test_df[:, :-3], test_df[:, -3:]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation="sigmoid"))
    # model.add(LSTM(32, input_shape=(9, 1), return_sequences=True, dropout=0.2))
    # model.add(LSTM(10, dropout=0.2))
    # model.add(Dense(3, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=1)

    train_predict = model.predict(X_train)
    train_score = mean_squared_error(Y_train, train_predict[:, :])
    print(train_score)

    test_prredict = model.predict(X_test)
    test_score = mean_squared_error(Y_test, test_prredict[:, :])
    print(test_score)


    # # define dataset
    # data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    # print(data)
    # # data split
    # n_test = 4
    # # model configs
    # cfg_list = sarima_configs()
    # # grid search
    # scores = grid_search(data, cfg_list, n_test)
    # print('done')
    # # list top 3 configs
    # for cfg, error in scores[:3]:
    #     print(cfg, error)

