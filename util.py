import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from datetime import datetime

# column headers: <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
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


def evaluate_test_set(model, test_env):
    obs = test_env.reset()
    for i in range(len(test_df)):
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        test_env.render(mode='verbose')
        if done:
            test_env.reset()

    plot_metrics(test_env.get_attr('metrics')[0])


def evaluate_train_set(model, train_env):
    print("Start testing on train set (for overfitting check")
    obs = train_env.reset()
    for i in range(6000):
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        train_env.render(mode='verbose')
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

    interval = TIME_FRAME[tf_type]

    times = df.Time.apply(lambda x: str(x)[-2:]).astype(int)
    indices = times % interval == 0
    df = df[indices]

    df.reset_index(drop=True, inplace=True)

    save_file(df, input_file, output_file)
    print("Convert data to {} is done.".format(tf_type))


def reformat_date_of_year(input_file, format='%Y-%m-%d', output_file=None):
    # format dayofyear field from int (20110103) to datetime format (2011-01-03)
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
    # format time field from int (85200) to datetime format (08:05:00)
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


def plot_data(input_file):
    print("Start plotting our data set")
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)

    df['logged_and_diffed'] = (np.log(df['Close']) - np.log(df['Close']).shift(1)) * 100
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


def plot_metrics(metrics):

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
        f.write("{:<25s}{:>5.2f}\n".format("Num step:", metrics['num_step'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Total win trades:", metrics['win_trades'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Total lose trades:", metrics['lose_trades'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Avg reward:", metrics['avg_reward'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Avg win value:", metrics['avg_win_value']))
        f.write("{:<25s}{:>5.2f}\n".format("Avg lose value:", metrics['avg_lose_value']))
        f.write("{:<25s}{:>5.2f}\n".format("Most profit trade win:", metrics['most_profit_trade'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Worst trade lose:", metrics['worst_trade'][-1]))
        f.write("{:<25s}{:>5.2f}\n".format("Highest net worth:", metrics['highest_net_worth']))
        f.write("{:<25s}{:>5.2f}\n".format("Lowest net worth:", metrics['lowest_net_worth']))
        f.write("{:<25s}{:>5.2f}\n".format("Win ratio:", metrics['win_trades'][-1] / (metrics['win_trades'][-1] + 1 + metrics['lose_trades'][-1])))
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
    print("Start encode time")
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)

    def encode(x):
        hour = int(x[0:2])
        minute = int(x[3:5])
        current_time = hour * 60 + minute
        encoded_time = current_time / (24 * 60)
        return encoded_time

    df['NormalizedTime'] = df['Time'].apply(lambda x: encode(x))
    save_file(df, input_file, output_file)
    print("encode time complete!")

def standardize_data(df):
    df["NormedClose"] = (np.log(df['Close']) - np.log(df['Close']).shift(1)) * 100
    df["Open"] = (np.log(df['Open']) - np.log(df['Open']).shift(1)) * 100
    df["High"] = (np.log(df['High']) - np.log(df['High']).shift(1)) * 100
    df["Low"] = (np.log(df['Low']) - np.log(df['Low']).shift(1)) * 100
    return df


if __name__ == '__main__':
    # convert_txt_to_csv("data/EURUSD_2011_2019.txt", "data/EURUSD.csv")
    # reduce_to_time_frame("./data/EURUSD.csv", 'm15', "./data/EURUSD_m15.csv")
    # reformat_date_of_year("./data/EURUSD_m15.csv")
    # reformat_time("./data/EURUSD_m15.csv")
    plot_data('./data/EURUSD_m15_train.csv')
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
    # split_dataset("./data/EURUSD_m15.csv", split_ratio=0.9)

    pass





