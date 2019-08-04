import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


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


def save_file(df, input_file, output_file=None):
    # utility function to help save the file in proper format
    if output_file is not None:
        df.to_csv(output_file, float_format='%.4f')
    else:
        df.to_csv(input_file, float_format='%.4f')


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

    df.reset_index(inplace=True, drop=True)

    interval = TIME_FRAME[tf_type]

    indices = np.arange(0, len(df), interval).tolist()

    df = df.iloc[indices].reset_index(drop=True)

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

    import time

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

    import time

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
    df['logged_and_diffed'] = np.log(df['Close']) - np.log(df['Close']).shift(1)

    plt.plot(df.index.values, df.logged_and_diffed.values)
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
    ax5.plot(metrics['num_step'], metrics['highest_networth'])
    plt.xlabel("highest_networth")
    ax5.xaxis.set_label_position('top')

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(metrics['num_step'], metrics['lowest_networth'])
    plt.xlabel("lowest_networth")
    ax6.xaxis.set_label_position('top')

    plt.show()


if __name__ == '__main__':
    # convert_txt_to_csv("data/EURUSD.txt", "data/EURUSD.csv")
    # reduce_to_time_frame("./data/EURUSD.csv", 'm15', "./data/EURUSD_m15.csv")
    # reformat_date_of_year("./data/EURUSD_m15.csv")
    # reformat_time("./data/EURUSD_m15.csv")
    # split_dataset("./data/EURUSD_m15.csv", split_ratio=0.85)
    # plot_data('./data/EURUSD_m15_test.csv')
    metrics = {"num_step": np.linspace(1, 10), "win_trades": np.linspace(1, 10), "lose_trades": np.linspace(1, 10), "avg_reward": np.linspace(1, 10), "most_profit_trade": np.linspace(1, 10),
                    "worst_trade": np.linspace(1, 10), "highest_networth": np.linspace(1, 10), "lowest_networth": np.linspace(1, 10)}

    plot_metrics(metrics)
    pass





