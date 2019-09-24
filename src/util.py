import plotly.graph_objects as go
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from datetime import datetime
from src.html_parser import ForexNewsParser

# column headers: <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
from sklearn.preprocessing import StandardScaler

TIME_FRAME = {
    "m1": 1,
    "m5": 5,
    "m15": 15,
    "m30": 30,
    "h1": 60,
    "h4": 240,
    'd1': 60*24,
    'w1': 60*24*7,
}

FULL_DATA_FILE = './data/EURUSD_m15.csv'
TRAIN_FILE = './data/EURUSD_m15_train.csv'
TEST_FILE = './data/EURUSD_m15_test.csv'


def evaluate_test_set(model, test_env, num_steps, mode='verbose'):
    """evaluate model on full test set"""
    obs = test_env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        test_env.render(mode=mode)
        if done:
            test_env.reset()

    plot_metrics(test_env.get_attr('metrics')[0])


def evaluate_train_set(model, train_env, num_steps, mode='verbose'):
    """evaluate model on train set"""
    print("Start testing on train set (for overfitting check")
    obs = train_env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        train_env.render(mode=mode)
        if done:
            train_env.reset()

    plot_metrics(train_env.get_attr('metrics')[0])


def save_file(df, input_file, output_file=None):
    # utility function to help save the file in proper format
    if output_file is not None:
        print("Saving output to: ", output_file)
        df.to_csv(output_file, float_format='%.5f')
    else:
        print("Saving output to: ", input_file)
        df.to_csv(input_file, float_format='%.5f')


def convert_txt_to_csv(input_file, output_file=None):
    """ convert file data from txt to csv"""
    print("Start convert: {} to csv".format(input_file))
    df = pd.read_csv(input_file,
                     sep=',',
                     header=None,
                     names=["Ticker", "DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])

    # the 0 row is redundant so we remove it manually
    df.drop(0, inplace=True)
    # drop useless columns
    df.drop(["Ticker", "Vol"], axis=1, inplace=True)

    save_file(df, input_file, output_file)
    print("Convert data to csv is done.")


def reduce_to_time_frame(input_file, tf_type, output_file=None):
    # convert the dataset to higher timeframe format, available options are: m5, m15, m30, h1, h4
    print("Start reduce time frame: to {}".format(tf_type))
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)
    # convert dataframe to numpy for faster processing speed
    df_array = df.to_numpy()

    interval = TIME_FRAME[tf_type]
    j = 0
    start = 1
    highest = -10
    lowest = 10

    for i in range(1, len(df)):
        j += 1
        # save lowest and highest price in INTERVAL we want
        highest = max(highest, df_array[i][3])
        lowest = min(lowest, df_array[i][4])

        # convert time to integer then use it to check that we hit the limit of desire timeframe
        time = int(str(df_array[i][1])[-2:])
        if time % interval == 0:
            # update row correspond to timeframe, row that divisible to timeframe's interval
            df.iat[i, 3] = highest
            df.iat[i, 4] = lowest
            df.iat[i, 2] = df.iat[start, 2]
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

    save_file(df, input_file, output_file)
    print("Convert data to {} is done.".format(tf_type))


def reformat_date_of_year(input_file, format='%Y-%m-%d', output_file=None):
    # format dayofyear field from int (20110103) to datetime format (2011-01-03)
    # WARNING: no longer used
    print("Start reformat datetime field")
    df = pd.read_csv(input_file,
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
    save_file(df, input_file, output_file)
    print("reformat datetime field to is done.")


def reformat_time(input_file, output_file=None):
    # format time field from int (0055) to datetime format (00:55)
    # WARNING: no longer used
    print("Start reformat Time field")
    df = pd.read_csv(input_file,
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
    save_file(df, input_file, output_file)
    print("reformat Time field to is done.")


def date2num(date):
    converter = mdates.strpdate2num('%YYYY-%mm-%dd')
    return converter(date)


def split_dataset(input_file, split_ratio=0.8, train_file_name=None, test_file_name=None):
    print("Start split dataset into train and test set")
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)

    split_point = int(split_ratio * len(df))

    df_train = df[:split_point]
    df_test = df[split_point:-1].reset_index(drop=True)

    if train_file_name is None:
        train_file_name = input_file[:-4] + "_train.csv"
    if test_file_name is None:
        test_file_name = input_file[:-4] + "_test.csv"

    save_file(df_train, None, train_file_name)
    save_file(df_test, None, test_file_name)
    print("split dataset complete!")


def heikin_ashi_candle(input_file, output_file=None):
    print("Start insert heikin ashi candle to our dataset")
    df = pd.read_csv(input_file,
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
    save_file(df, input_file, output_file)
    print("Insert keikin ashi candle is done.")
    return df


def plot_data(input_file):
    print("Start plotting our data set")
    df = pd.read_csv(input_file,
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

    with open("./logs/metrics.txt", 'a+') as f:
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


def merge_data_2012_2018():
    data_2012 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2012.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2013 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2013.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2014 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2014.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2015 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2015.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2016 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2016.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2017 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2017.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2018 = pd.read_csv('./data/DAT_MT_EURUSD_M1_2018.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])

    data_2012_2018 = data_2012.append(data_2013)
    data_2012_2018 = data_2012_2018.append(data_2014)
    data_2012_2018 = data_2012_2018.append(data_2015)
    data_2012_2018 = data_2012_2018.append(data_2016)
    data_2012_2018 = data_2012_2018.append(data_2017)
    data_2012_2018 = data_2012_2018.append(data_2018)

    data_2012_2018.drop(["Vol"], axis=1, inplace=True)
    data_2012_2018.reset_index(inplace=True, drop=True)

    save_file(data_2012_2018, input_file="./data/EURUSD.csv")


def merge_data_2019():
    data_2019_01 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201901.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_02 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201902.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_03 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201903.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_04 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201904.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_05 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201905.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_06 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201906.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    data_2019_07 = pd.read_csv('./data/DAT_MT_EURUSD_M1_201907.csv',
                            names=["DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])

    data_2019 = data_2019_01.append(data_2019_02)
    data_2019 = data_2019.append(data_2019_03)
    data_2019 = data_2019.append(data_2019_04)
    data_2019 = data_2019.append(data_2019_05)
    data_2019 = data_2019.append(data_2019_06)
    data_2019 = data_2019.append(data_2019_07)

    data_2019.drop(["Vol"], axis=1, inplace=True)
    data_2019.reset_index(inplace=True, drop=True)

    save_file(data_2019, input_file="./data/EURUSD_2019.csv")


def encode_time(input_file, output_file=None):
    # encode time feature to floating values, range [0-1]
    print("Start encode time")
    df = pd.read_csv(input_file,
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

    save_file(df, input_file, output_file)
    print("encode time complete!")


def standardize_data(df, method='log_and_diff'):
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


def get_episode(df):
    # get indices of episode in data frame, each episode is one week data
    def get_end_week_indices(row):
        # get index of one week trading session,
        # weekend at day = 4 (thursday and time = 16:45
        if row.DayOfWeek == 4 and row.Time == "16:45":
            return row.name
        return -1

    indices = df.apply(lambda x: get_end_week_indices(x), axis=1)
    indices = indices[indices != -1].to_numpy()
    true_indices = [(0, indices[0])]

    # append start week index to make a list of tuple contains (start, end) indices
    for i in range(1, len(indices)):
        true_indices.append((indices[i-1] + 1, indices[i]))

    return true_indices


def augmented_dickey_fuller_test(input_file):
    from statsmodels.tsa.stattools import adfuller
    print("Start augmented_dickey_fuller_test on data set")
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)

    df['logged_and_diffed'] = ((df['Close']) - ((df['Close']).shift(1))) * 100
    result = adfuller(df['logged_and_diffed'].values[1:], autolag="AIC")
    print("test result: ", result)


def insert_economical_news_feature(input_file, html_file, output_file=None):
    """
    insert feature indicates that agent is in a high risk state
    (ecomical news is going to be announced)
    :param output_file: (string) output file name and path
    :param html_file: (string) path to html file contains economic calendar
    :param input_file: (string) path to input csv file
    :return: None
    """
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)
    df["HighRiskTime"] = 0
    # read and parse the html to get dates have economical news announced
    print("Start insert economical news feature to dataset")
    with open(html_file, 'r', encoding='utf-8') as f:
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

    save_file(df, input_file, output_file)
    print("insert successfully")


def show_candles_chart(input_file, start, period=100, candle_type="normal"):
    """
    show OHLC price in candle shape
    :param input_file: (string) file name we want to read from
    :param start: (int) index of specific point in time we want to show
    :param period: (int) we normally want to see candles from start to start + period of time
    :return: None
    """
    end = start + period
    df = pd.read_csv(input_file)
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


def data_exploration(input_file):
    """
    perform EDA on input file
    :param input_file: csv file contains price history
    :return: None
    """
    df = pd.read_csv(input_file, sep=',', index_col=0)
    price_diff = np.abs(df.HeikinOpen - df.HeikinClose)
    print(price_diff.quantile([0.3, 0.6]))


def categorize_data(input_file, output_file=None):
    """
    categorize price to 7 types: side way, small up, medium up, big up, small down, medium down, big down
    NOTE: we could add more type to make our data representation more precise
    :param input_file: csv file contains price history
    :return: None
    """
    print("Start categorize data")
    df = pd.read_csv(input_file, sep=',', index_col=0)
    df['PriceDiff'] = (df.HeikinClose - df.HeikinOpen) * 10000
    df["CandleType"] = 1                                # side way

    df["CandleType"][(1.5 <= df.PriceDiff) & (df.PriceDiff <= 2.6)] = 2    # small up
    df["CandleType"][(2.7 <= df.PriceDiff) & (df.PriceDiff <= 5.3)] = 3    # medium up
    df["CandleType"][df.PriceDiff >= 5.4] = 4           # big up

    df["CandleType"][(-1.5 >= df.PriceDiff) & (df.PriceDiff >= -2.6)] = 5  # small down
    df["CandleType"][(-2.7 >= df.PriceDiff) & (df.PriceDiff >= -5.3)] = 6    # medium down
    df["CandleType"][df.PriceDiff <= -5.4] = 7          # big down

    df.drop(["PriceDiff"], axis=1, inplace=True)
    save_file(df, input_file, output_file)
    print("Categorize data's finished")


if __name__ == '__main__':

    # data_exploration("./data/EURUSD_m15_train.csv")

    # convert_txt_to_csv("data/EURUSD_2011_2019.txt", "data/EURUSD.csv")
    # reduce_to_time_frame("./data/EURUSD.c sv", 'm15', "./data/EURUSD_m15.csv")

    # plot_data('./data/EURUSD_m15_train.csv')
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

    # encode_time("./data/EURUSD_m15.csv")

    # augmented_dickey_fuller_test('./data/EURUSD_m15_train.csv')
    # insert_economical_news_feature(FULL_DATA_FILE, "./data/Economic Calendar - Investing.com.html")
    # heikin_ashi_candle(FULL_DATA_FILE)
    categorize_data(FULL_DATA_FILE)
    split_dataset(FULL_DATA_FILE, split_ratio=0.9)
    #
    # show_candles_chart(TRAIN_FILE, 16000, 150, candle_type='normal')
    # show_candles_chart(TRAIN_FILE, 16000, 150, candle_type="heikin")
    pass





