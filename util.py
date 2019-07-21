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

def convert_txt_to_csv(input_file, output_file=None):
    print("Start convert: {} to csv".format(input_file))
    df = pd.read_csv(input_file,
                     sep=',',
                     header=None,
                     names=["Ticker", "DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])

    df.drop(["Ticker", "Vol"], axis=1, inplace=True)

    if output_file is not None:
        df.to_csv(output_file, float_format='%.4f')
    else:
        df.to_csv(input_file, float_format='%.4f')
    print("Convert data to csv is done.")
# convert_txt_to_csv("data/EURUSD.txt", "data/EURUSD.csv")

def reduce_to_time_frame(input_file, tf_type, output_file=None):
    print("Start reduce time frame: to {}".format(tf_type))
    df = pd.read_csv(input_file,
                     sep=',', index_col=0)

    df.reset_index(inplace=True, drop=True)

    interval = TIME_FRAME[tf_type]

    indices = np.arange(0, len(df), interval).tolist()

    df = df.iloc[indices].reset_index(drop=True)

    if output_file is not None:
        df.to_csv(output_file, float_format='%.4f')
    else:
        df.to_csv(input_file, float_format='%.4f')
    print("Convert data to {} is done.".format(tf_type))

reduce_to_time_frame("./data/EURUSD.csv", 'm15', "./data/EURUSD_m15.csv")
