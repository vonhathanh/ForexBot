import pandas as pd


# column headers: <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
def convert_txt_to_csv(input_file, output_file=None):
    print("Start convert: {} to csv".format(input_file))
    df = pd.read_csv(input_file, sep=',', header=None, names=["Ticker", "DayOfYear", "Time", "Open", 'High', 'Low', 'Close', 'Vol'])
    df.drop(["Ticker", "Vol"], axis=1, inplace=True)
    if output_file is not None:
        df.to_csv(output_file)
    else:
        df.to_csv(input_file)
    print("Convert data to csv is done.")


convert_txt_to_csv("data/EURUSD.txt", "data/EURUSD.csv")