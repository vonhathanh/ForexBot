import os

# a trading session is a fragment of training data
# when we've finished training agent in a session, we move on to another one (different fragment)
# this variable determined maximum time allowed in a session
MAX_TRADING_SESSION = 2000

# 1 lot in forex equal to 100000 usd
LOT_SIZE = 100000

# amount when buy, sell
AMOUNT = 0.5

# look back window, agent will use this to know what happended in the past 8 time frame
WINDOW_SIZE = 24

# comission when we making a trade
COMISSION = 0.0002

INITIAL_BALANCE = 100000

# this constant is used to normalize agent's balance to range [-1, 1]
# currently we don't have a more sophisticated method to do that so we just stick with this one
BALANCE_NORM_FACTOR = 500000
# action enum, openai gym don't accept enum class so we just use constants here
HOLD = 0
BUY = 1
SELL = 2
CLOSE = 3
CLOSE_AND_BUY = 4
CLOSE_AND_SELL = 5

ACTIONS = {HOLD: "hold",
           BUY: "buy",
           SELL: "sell",
           CLOSE: "close",
           CLOSE_AND_BUY: "close_and_buy",
           CLOSE_AND_SELL: "close_and_sell"}

# column headers: <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>

SRC_DIR = os.path.dirname(__file__)


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

RAW_TXT_FILE = os.path.join(SRC_DIR, "..", "data", "EURUSD.txt")
RAW_CSV_FILE = os.path.join(SRC_DIR, "..", "data", "EURUSD.csv")
FULL_DATA_FILE_m15 = os.path.join(SRC_DIR, "..", "data", "EURUSD_m15.csv")
FULL_DATA_FILE_h1 = os.path.join(SRC_DIR, "..", "data", "EURUSD_h1.csv")

TRAIN_FILE_m15 = os.path.join(SRC_DIR, "..", "data", "EURUSD_m15_train.csv")
TRAIN_FILE_h1 = os.path.join(SRC_DIR, "..", "data", "EURUSD_h1_train.csv")
TEST_FILE_m15 = os.path.join(SRC_DIR, "..", "data", "EURUSD_m15_test.csv")
TEST_FILE_h1 = os.path.join(SRC_DIR, "..", "data", "EURUSD_h1_test.csv")

EMBEDDED_WEIGHT_FILE = os.path.join(SRC_DIR, "..", "models", "embedding_weights.pkl")