



































def merge_data_2012_2018():
    """
    One-time used function
    :return:
    """
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

    save_file(data_2012_2018, original_file="./data/EURUSD.csv")


def merge_data_2019():
    """
    One-time used function
    :return:
   """
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

    save_file(data_2019, original_file="./data/EURUSD_2019.csv")
