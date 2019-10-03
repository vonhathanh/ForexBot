import plotly.graph_objects as go
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
import os

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from datetime import datetime
from html_parser import ForexNewsParser
from sklearn.preprocessing import StandardScaler
from constants import *
from mpl_toolkits import mplot3d


def evaluate_test_set(model, test_env, num_steps, mode='verbose'):
    """
    evaluate model on full test set
    :param model: trained stable baseline model
    :param test_env: (DummyVecEnv) environment for testing model
    :param num_steps: (Int) number of time step we will traverse
    :param mode: (Str) render mode, verbose = text only, human = display graphics
    :return: None
    """
    obs = test_env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        test_env.render(mode=mode)
        if done:
            test_env.reset()

    plot_metrics(test_env.get_attr('metrics')[0])


def evaluate_train_set(model, train_env, num_steps, mode='verbose'):
    """
    evaluate model on train set
    :param model: trained stable baseline model
    :param train_env: (DummyVecEnv) environment for testing model
    :param num_steps: (Int) number of time step we will traverse
    :param mode: (Str) render mode, verbose = text only, human = display graphics
    :return:
    """
    print("Start testing on train set (for overfitting check")
    obs = train_env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        train_env.render(mode=mode)
        if done:
            train_env.reset()

    plot_metrics(train_env.get_attr('metrics')[0])


def save_file(df, original_file, output_file_path=None):
    """
    utility function to help save the file in proper format
    :param df: (DataFrame) data we want to save
    :param original_file: (Str) use when we want to override the original file
    :param output_file_path: Str) relative path leads to where we want the data to be saved in different file
    :return: None
    """
    if output_file_path is not None:
        print("Saving output to: ", output_file_path)
        df.to_csv(output_file_path, float_format='%.5f')
    else:
        print("Saving output to: ", original_file)
        df.to_csv(original_file, float_format='%.5f')


def convert_txt_to_csv(input_file_path, output_file_path=None):
    """
    convert file data from txt to csv
    :param input_file_path: (Str) relative path lead to data file
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """
    print("Start convert: {} to csv".format(input_file_path))
    df = pd.read_csv(input_file_path,
                     sep=',',
                     header=None,
                     names=["Ticker", "DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])

    # the 0 row is redundant so we remove it manually
    df.drop(0, inplace=True)
    # drop useless columns
    df.drop(["Ticker", "Vol"], axis=1, inplace=True)

    save_file(df, input_file_path, output_file_path)
    print("Convert data to csv is done.")


def reduce_to_time_frame(input_file_path, tf_type, output_file_path=None):
    """
    convert the dataset to higher timeframe format,
    available options are: m5, m15, m30, h1, h4
    :param input_file_path: (Str) relative path leads to data file
    :param tf_type: (Str) time frame name
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return:
    """
    print("Start reduce time frame: to {}".format(tf_type))
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)
    # convert dataframe to numpy for faster processing speed
    df_array = df.to_numpy()

    interval = TIME_FRAME[tf_type]
    j = 0
    start = 1
    highest = -10
    lowest = 10
    open = df.columns.get_loc("Open")
    high = df.columns.get_loc("High")
    low = df.columns.get_loc("Low")

    for i in range(1, len(df)):
        j += 1
        # save lowest and highest price in INTERVAL we want
        highest = max(highest, df_array[i][high])
        lowest = min(lowest, df_array[i][low])

        # convert time to integer then use it to check that we hit the limit of desire timeframe
        time = int(str(df_array[i][1])[-2:])
        if time % interval == 0:
            # update row correspond to timeframe, row that divisible to timeframe's interval
            df.iat[i, high] = highest
            df.iat[i, low] = lowest
            df.iat[i, open] = df.iat[start, open]
            # reset variables
            j = 0
            highest = -10
            lowest = 10
            start = i+1

        if i % 5000 == 0:
            print("current step: ", i)

    # get indices of rows we updated
    times = df.Time.apply(lambda x: str(x)[-2:]).astype(int)
    indices = times % interval == 0
    # create new dataframe, using only indices we have collected.
    df = df[indices]

    df.reset_index(drop=True, inplace=True)

    save_file(df, input_file_path, output_file_path)
    print("Convert data to {} is done.".format(tf_type))


def reformat_date_of_year(input_file_path, format='%Y-%m-%d', output_file_path=None):
    """
    format dayofyear field from int (20110103) to datetime format (2011-01-03)
    WARNING: no longer used
    :param input_file_path: (Str) relative path leads to data file
    :param format: (Str) date format, example: '%Y-%m-%d'
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """

    print("Start reformat datetime field")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)

    def get_plit_date(date):
        day = date % 100
        date //= 100
        month = date % 100
        year = date // 100
        return str(year) + '-' + str(month) + '-' + str(day)

    start = time.time()
    df['DayOfYear'] = pd.to_datetime(df['DayOfYear'].apply(lambda x: get_plit_date(x)), format=format)
    print(df['DayOfYear'])
    print("time cost: ", time.time() - start)
    save_file(df, input_file_path, output_file_path)
    print("reformat datetime field to is done.")


def reformat_time(input_file_path, output_file_path=None):
    """
    format time field from int (0055) to datetime format (00:55)
    WARNING: no longer used
    :param input_file_path: (Str) relative path leads to data file
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """

    print("Start reformat Time field")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)

    def padd_zero(time):
        zero_to_fill = 4 - len(time)
        time = '0'*zero_to_fill + time
        time = time[:2] + ':' + time[2:]
        return time

    start = time.time()
    df['Time'] = df['Time'].apply(lambda x: x // 100)
    df['Time'] = df['Time'].apply(lambda x: padd_zero(str(x)))

    print("time cost: ", time.time() - start)
    save_file(df, input_file_path, output_file_path)
    print("reformat Time field to is done.")


def date2num(date):
    """
    Convert date from pandas data type to string type
    :param date: Pandas date type
    :return: Data formatted in string
    """
    converter = mdates.strpdate2num('%YYYY-%mm-%dd')
    return converter(date)


def split_dataset(input_file_path, split_ratio=0.8, train_file_name=None, test_file_name=None):
    """
    Split the dataset into train data and testing data
    :param input_file_path: (Str) relative path leads to data file
    :param split_ratio: (Float) ratio between train set and test set, range from 0.0 to 1.0
           0.6 means 60% data is training data and 40% is testing data
    :param train_file_name: (Str) file name of training data after save
    :param test_file_name:(Str) file name of testing data after save
    :return: None
    """
    print("Start split dataset into train and test set")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)

    split_point = int(split_ratio * len(df))

    df_train = df[:split_point]
    df_test = df[split_point:-1].reset_index(drop=True)

    if train_file_name is None:
        train_file_name = input_file_path[:-4] + "_train.csv"
    if test_file_name is None:
        test_file_name = input_file_path[:-4] + "_test.csv"

    save_file(df_train, None, train_file_name)
    save_file(df_test, None, test_file_name)
    print("split dataset complete!")


def insert_heikin_ashi_candle_feature(input_file_path, output_file_path=None):
    """
    Add heikin ashi candle features to input file
    :param input_file_path: (Str) relative path leads to input data file
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """
    print("Start insert heikin ashi candle to our dataset")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)
    df['HeikinClose'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HeikinOpen'] = 0.0
    df['HeikinHigh'] = 0.0
    df['HeikinLow'] = 0.0

    df.iat[0, 13] = df['Open'].iloc[0]
    for i in range(1, len(df)):
        df.iat[i, 13] = (df.iat[i - 1, 13] + df.iat[i - 1, 12]) / 2

    df['HeikinHigh'] = df.loc[:, ['HeikinOpen', 'HeikinClose']].join(df['High']).max(axis=1)

    df['HeikinLow'] = df.loc[:, ['HeikinOpen', 'HeikinClose']].join(df['Low']).min(axis=1)
    save_file(df, input_file_path, output_file_path)
    print("Insert keikin ashi candle is done.")
    return df


def plot_data(input_file_path):
    """
    Plot data for easier visualization
    :param input_file_path: (Str) relative path leads to input data file
    :return: None
    """
    print("Start plotting our data set")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)

    df['logged_and_diffed'] =(df['Close'] - df['Close'].shift(1)) * 100
    print("quantile: ", df["logged_and_diffed"].quantile(0.9))
    print(round(df["logged_and_diffed"].mean(), 5))
    df['z_norm'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()

    plt.subplot(3, 1, 1)
    plt.plot(df.index.values, df.logged_and_diffed.values, label='logged_price')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df.index.values, df.z_norm.values, label='znorm_price')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df.index.values, df.Close.values, label='normal_price')
    plt.legend()

    plt.show()


def plot_metrics(metric):
    """
    Plot the metrics to evaluate agent's performance after training, testing
    And write the result to ../logs/metrics.txt
    :param metric: Dictionary contains information about agent trading hisotry
    :return: None
    """
    metrics = metric.metrics
    ax1 = plt.subplot(2,3,1)
    ax1.plot(metrics['num_step'], metrics['win_trades'], label='win trades')
    ax1.plot(metrics['num_step'], metrics['lose_trades'], label='lose trades')
    plt.legend(loc='upper left')
    plt.xlabel("win vs lose trade")
    ax1.xaxis.set_label_position('top')

    ax2 = plt.subplot(2,3,2)
    ax2.plot(metrics['num_step'], metrics['avg_reward'])
    plt.xlabel("avg reward")
    ax2.xaxis.set_label_position('top')

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(metrics['num_step'], metrics['most_profit_trade'])
    plt.xlabel("most_profit_trade")
    ax3.xaxis.set_label_position('top')

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(metrics['num_step'], metrics['worst_trade'])
    plt.xlabel("worst_trade")
    ax4.xaxis.set_label_position('top')

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(metrics['num_step'], metrics['net_worth'])
    plt.xlabel("networth")
    ax5.xaxis.set_label_position('top')

    plt.show()

    with open( os.path.join(SRC_DIR, "..", "logs", "metrics.txt"), 'a+') as f:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d, %H:%M:%S")
        f.write(date + "\n")
        f.write("{:<25s}{:>5.2f}\n".format("Num step:", metric.num_step))
        f.write("{:<25s}{:>5.2f}\n".format("Total win trades:", metric.win_trades))
        f.write("{:<25s}{:>5.2f}\n".format("Total lose trades:", metric.lose_trades))
        f.write("{:<25s}{:>5.2f}\n".format("Avg reward:", metric.avg_reward / metric.num_step))
        f.write("{:<25s}{:>5.2f}\n".format("Avg win value:", metric.avg_win_value))
        f.write("{:<25s}{:>5.2f}\n".format("Avg lose value:", metric.avg_lose_value))
        f.write("{:<25s}{:>5.2f}\n".format("Most profit trade win:", metric.most_profit_trade))
        f.write("{:<25s}{:>5.2f}\n".format("Worst trade lose:", metric.worst_trade))
        f.write("{:<25s}{:>5.2f}\n".format("Highest net worth:", metric.highest_net_worth))
        f.write("{:<25s}{:>5.2f}\n".format("Lowest net worth:", metric.lowest_net_worth))
        f.write("{:<25s}{:>5.2f}\n".format("Win ratio:", metric.win_trades / (metric.win_trades + 1 + metric.lose_trades)))
        f.write("-" * 80 + "\n")


def encode_time(input_file_path, output_file_path=None):
    """
    encode time feature to floating values, range [0-1]
    :param input_file_path: (Str) relative path leads to input data file
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """
    print("Start encode time")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)
    df["DayOfWeek"] = pd.to_datetime(df['DayOfYear'])
    df["DayOfWeek"] = df.DayOfWeek.apply(lambda x: x.dayofweek)

    def encode(x):
        hour = int(x[0:2])
        minute = int(x[3:5])
        current_time = hour * 60 + minute
        return current_time

    df['TimeInMinute'] = df['Time'].apply(lambda x: encode(x))
    df['TimeEncodedX'] = np.sin(2 * np.pi * df.TimeInMinute / 1425)
    df['TimeEncodedY'] = np.cos(2 * np.pi * df.TimeInMinute / 1425)
    df['DayEncodedX'] = np.sin(2 * np.pi * df.DayOfWeek / 5)
    df['DayEncodedY'] = np.cos(2 * np.pi * df.DayOfWeek / 5)
    df.drop(["TimeInMinute"], axis=1, inplace=True)

    save_file(df, input_file_path, output_file_path)
    print("encode time complete!")


def standardize_data(df, method='log_and_diff'):
    """
    normalize data using different methods
    :param df: (Dataframe) input data to be processed
    :param method: (Str) specify which method will be used to normalizing
    :return: (DataFrame) data after normalizing
    """
    if method == 'log_and_diff':
        df["NormedClose"] = (df['HeikinClose'] - df['HeikinClose'].shift(1)) * 100
        df["Open"] = (df['HeikinOpen'] - df['HeikinOpen'].shift(1)) * 100
        df["High"] = (df['HeikinHigh'] - df['HeikinHigh'].shift(1)) * 100
        df["Low"] = (df['HeikinLow'] - df['HeikinLow'].shift(1)) * 100
    elif method == 'z_norm':
        scaler = StandardScaler()
        df["NormedClose"] = scaler.fit_transform(df['Close'].to_numpy().reshape((-1, 1)))
        df["Open"] = scaler.fit_transform(df['Open'].to_numpy().reshape((-1, 1)))
        df["High"] = scaler.fit_transform(df['High'].to_numpy().reshape((-1, 1)))
        df["Low"] = scaler.fit_transform(df['Low'].to_numpy().reshape((-1, 1)))
    else:
        raise RuntimeError("Unknown normalization method, valid methods are: log_and_diff, z_norm")
    return df


def get_episode(df_m15, df_h1):
    """
    get indices of episode in data frame, each episode is one week data
    :param df_h1: (Dataframe) input data to be processed, timeframe: 1 hour
    :param df_m15: (Dataframe) input data to be processed, timeframe: 15 mins
    :return: (Array of Int) indices of time step where trading start (usually start of the week)
    """
    def get_end_week_indices(row):
        # get index of one week trading session,
        # weekend at day = 4 (thursday and time = 16:45
        if row.DayOfWeek == 4 and row.Time == "16:45":
            return row.name
        return -1

    indices = df_m15.apply(lambda x: get_end_week_indices(x), axis=1)
    indices = indices[indices != -1].to_numpy()

    # append start week index to make a list of tuple contains (start, end) indices
    episode_indices = [(0, indices[0])]
    for i in range(1, len(indices)):
        episode_indices.append((indices[i-1] + 1, indices[i]))

    # get h1 indices
    h1_indices = []
    i = 0
    for index_tuple in episode_indices:
        start_index = index_tuple[0] + WINDOW_SIZE
        # update start_index, makes it point to start of hour and new index must <= old start_index
        # because if index is greater than it's old value, it is future data
        for k in range(4):
            hour = df_m15.Time[start_index - k]
            if hour[-2:] == "00":
                start_index = start_index - k
                break
        # loop over data in hour frame to get corresponding index to start_index in 15 mins time frame data
        for _ in range(i, len(df_h1)):
            # compare values that is combined from day and hour to find exact index we want
            day = df_h1.DayOfYear[i]
            hour = df_h1.Time[i]
            full_day_time_h1 = day + ":" + hour

            day = df_m15.DayOfYear[start_index]
            hour = df_m15.Time[start_index]
            full_day_time_m15 = day + ":" + hour

            # append whatever index we found that match our condition
            if full_day_time_h1 == full_day_time_m15:
                h1_indices.append(i)
                break
            else:
                i += 1

    return episode_indices, h1_indices


def augmented_dickey_fuller_test(input_file_path):
    """
    perfrom ADF test to ensure data has the properties we want
    :param input_file_path: (Str) relative path leads to input data file
    :return: None
    """
    from statsmodels.tsa.stattools import adfuller
    print("Start augmented_dickey_fuller_test on data set")
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)

    df['logged_and_diffed'] = ((df['Close']) - ((df['Close']).shift(1))) * 100
    result = adfuller(df['logged_and_diffed'].values[1:], autolag="AIC")
    print("test result: ", result)


def insert_economical_news_feature(input_file_path, html_file_path, output_file_path=None):
    """
    insert feature indicates that agent is in a high risk state
    (ecomical news is going to be announced)
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :param html_file_path: (Str) relative path to html file contains economic calendar
    :param input_file_path: Str) relative path leads to input data file
    :return: None
    """
    df = pd.read_csv(input_file_path,
                     sep=',', index_col=0)
    df["HighRiskTime"] = 0
    # read and parse the html to get dates have economical news announced
    print("Start insert economical news feature to dataset")
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_raw = f.read()
    parser = ForexNewsParser()
    parser.feed(html_raw)
    # sort dates in ascending order, replace "/" by "." to match with training data
    dates = sorted(parser.dates)
    dates = list(map(lambda x: x.replace("/", ".")[:-3], dates))

    # if both day and time match between the dataset and calendar
    # we set highrisktime = 1
    t = 0
    for i in range(len(df)):
        # date format is yyyy.dd.mm  hh:mm, we split it to day and time
        data = dates[t].split(" ")
        day = data[0]
        time = data[1]
        # compare day time between two data sets

        if df.iat[i, 0] == day and df.iat[i, 1] == time:
            print(df.iat[i, 0], " ", df.iat[i, 1])
            df.HighRiskTime[i-1:i+5] = 1
            t += 1
        # increase economical calender index by one if we have passed the day event occur
        elif df.iat[i, 0] > day:
            t += 1

    save_file(df, input_file_path, output_file_path)
    print("insert successfully")


def show_candles_chart(input_file_path, start, period=100, candle_type="normal"):
    """
    show OHLC price in candle shape
    :param candle_type: (string) specify candle type: "normal" or "heiken"
    :param input_file_path: (string) relative file path leads to data file
    :param start: (int) index of specific point in time we want to show
    :param period: (int) we normally want to see candles from start to start + period of time
    :return: None
    """
    end = start + period
    df = pd.read_csv(input_file_path)
    df["DayOfYear"] = df["DayOfYear"] + " " + df["Time"] + ":00"
    df["DayOfYear"] = df["DayOfYear"].apply(lambda x: str(x).replace(".", "-"))
    if candle_type == "normal":
        fig = go.Figure(data=[go.Candlestick(x=df["DayOfYear"][start: end],
                                             open=df['Open'][start: end],
                                             high=df['High'][start: end],
                                             low=df['Low'][start: end],
                                             close=df['Close'][start: end])])
    else:
        fig = go.Figure(data=[go.Candlestick(x=df["DayOfYear"][start: end],
                                             open=df['HeikinOpen'][start: end],
                                             high=df['HeikinHigh'][start: end],
                                             low=df['HeikinLow'][start: end],
                                             close=df['HeikinClose'][start: end])])

    fig.show()


def data_exploration(input_file_path):
    """
    perform EDA on input file
    :param input_file_path: (Str) relative file path leads to csv file contains price history
    :return: None
    """
    df = pd.read_csv(input_file_path, sep=',', index_col=0)
    price_diff = np.abs(df.HeikinOpen - df.HeikinClose)
    print(price_diff.quantile([0.3, 0.6]))


def categorize_price_data(input_file_path, output_file_path=None):
    """
    categorize price to 7 types: side way, small up, medium up, big up, small down, medium down, big down
    NOTE: we could add more type to make our data representation more precise
    :param input_file_path: (Str) relative file path leads to data file we want process
    :param output_file_path: (Str) relative path lead to where we want the new data to be saved
    :return: None
    """
    print("Start categorize data")
    df = pd.read_csv(input_file_path, sep=',', index_col=0)
    df['PriceDiff'] = (df.HeikinClose - df.HeikinOpen) * 10000
    df["CandleType"] = 1                                # side way

    df["CandleType"][(1.5 <= df.PriceDiff) & (df.PriceDiff <= 2.6)] = 2    # small up
    df["CandleType"][(2.7 <= df.PriceDiff) & (df.PriceDiff <= 5.3)] = 3    # medium up
    df["CandleType"][df.PriceDiff >= 5.4] = 4           # big up

    df["CandleType"][(-1.5 >= df.PriceDiff) & (df.PriceDiff >= -2.6)] = 5  # small down
    df["CandleType"][(-2.7 >= df.PriceDiff) & (df.PriceDiff >= -5.3)] = 6    # medium down
    df["CandleType"][df.PriceDiff <= -5.4] = 7          # big down

    df.drop(["PriceDiff"], axis=1, inplace=True)
    save_file(df, input_file_path, output_file_path)
    print("Categorize data's finished")


def embedding_feature(input_file_path, output_file_path=EMBEDDED_WEIGHT_FILE, embedding_size=2):
    """
    embedding input feature into vector space
    :param input_file_path: (Str) relative path to csv file contains input feature
    :param embedding_size: (Int) number of dimensions that our embedding vector will have
    :param output_file_path: (Str) relative path to save file after embedding data, if None, save it to input file
    :return: None
    """
    df = pd.read_csv(input_file_path, sep=',', index_col=0)
    df.CandleType = (df.CandleType - 1).astype(int)

    model = Sequential()
    model.add(Embedding(input_dim=7, output_dim=embedding_size, input_length=1, name="embedding"))
    model.add(Flatten())
    model.add(Dense(25, activation="relu"))
    # model.add(Dense(15, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(x=df.CandleType.values, y=(df.HeikinClose - df.HeikinOpen).values * 10000, epochs=2, batch_size=8)
    model.summary()

    layer = model.get_layer('embedding')
    output_embeddings = layer.get_weights()
    print("output embedding weights: ", output_embeddings)
    with open(output_file_path, 'wb') as file:
        pickle.dump(output_embeddings[0], file, protocol=3)
    print("dumped embedding weights to: ", EMBEDDED_WEIGHT_FILE)


def visualize_embedding(input_file_path=EMBEDDED_WEIGHT_FILE):
    """
    display the embedding weights in 3d-4d plot
    :param input_file_path: (Str) relative path to file contains embedding weights
    :return: None
    """
    with open(input_file_path, 'rb') as file:
        weights = pickle.load(file)
    print("weights: ", weights)
    labels = np.arange(1, 8)
    fig = plt.figure(figsize=[20, 20])
    ax = fig.add_subplot(111)

    for i in range(len(labels)):
        ax.scatter(weights[i, 0], weights[i, 1], color='b')
        ax.text(weights[i, 0], weights[i, 1], str(labels[i]), size=20, zorder=1, color='k')

    ax.set_xlabel('Embedding 1')
    ax.set_ylabel('Embedding 2')
    plt.show()


def add_embedded_feature_to_dataset(input_csv_file, input_weights_file=EMBEDDED_WEIGHT_FILE, output_file=None):
    """
    insert embedded weights to our dataset
    :param input_csv_file: (Str) relative path to file contains dataset
    :param input_weights_file: (Str) relative path to file contains embedding weights
    :param output_file: (Str) relative path to save csv file after add embedded weights, if None, save it to input file
    :return: None
    """
    print("Start add embedded feature to dataset")
    df = pd.read_csv(input_csv_file, sep=",", index_col=0)
    with open(input_weights_file, "rb") as file:
        weights = pickle.load(file)

    candles = df.CandleType.values
    candle_to_weights = np.zeros((len(candles), 2))
    for i in range(len(candles)):
        candle_to_weights[i] = weights[candles[i] - 1]

    df["CandleEmbededX"] = candle_to_weights[:, 0]
    df["CandleEmbededY"] = candle_to_weights[:, 1]
    print("Add embeded feature has finished")
    save_file(df, input_csv_file)


if __name__ == '__main__':
    # data_exploration(TRAIN_FILE)
    #
    # convert_txt_to_csv(RAW_TXT_FILE, RAW_CSV_FILE)
    # reduce_to_time_frame(TRAIN_FILE_m15, 'h1', TRAIN_FILE_h1)
    # reduce_to_time_frame(TEST_FILE_m15, 'h1', TEST_FILE_h1)
    #
    # plot_data(TRAIN_FILE)
    # metrics = {"num_step": np.linspace(1, 10),
    #            "win_trades": np.linspace(1, 10),
    #            "lose_trades": np.linspace(1, 10),
    #            "avg_reward": np.linspace(1, 10),
    #            "most_profit_trade": np.linspace(1, 10),
    #            "worst_trade": np.linspace(1, 10),
    #            "highest_networth": np.linspace(1, 10),
    #            "lowest_networth": np.linspace(1, 10)}
    #
    # plot_metrics(metrics)

    # encode_time(FULL_DATA_FILE_m15)
    #
    # augmented_dickey_fuller_test(TRAIN_FILE)
    # insert_economical_news_feature(FULL_DATA_FILE_m15, "./data/Economic Calendar - Investing.com.html")
    # insert_heikin_ashi_candle_feature(TRAIN_FILE_h1)
    # insert_heikin_ashi_candle_feature(TEST_FILE_h1)
    #
    show_candles_chart(TRAIN_FILE_m15, 16000, 150, candle_type='heikin')
    show_candles_chart(TRAIN_FILE_h1, 4000, 150//4, candle_type="heikin")

    # categorize_price_data(FULL_DATA_FILE_m15)
    # embedding_feature(FULL_DATA_FILE_m15)
    # visualize_embedding()
    # add_embedded_feature_to_dataset(FULL_DATA_FILE_m15)
    # split_dataset(FULL_DATA_FILE_m15, split_ratio=0.9)

    pass





