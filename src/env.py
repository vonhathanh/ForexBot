import gym
import numpy as np

from trading_graph import StockTradingGraph
from gym import spaces
from util import standardize_data, get_episode
from metrics import Metric

# a trading session is a fragment of training data
# when we've finished training agent in a session, we move on to another one (different fragment)
# this variable determined maximum time allowed in a session
MAX_TRADING_SESSION = 2000

# 1 lot in forex equal to 100000 usd
LOT_SIZE = 100000

# amount when buy, sell
AMOUNT = 0.5

# look back window, agent will use this to know what happended in the past 8 time frame
WINDOW_SIZE = 48

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


# env that use lstm architecture to train the model
class LSTM_Env(gym.Env):

    def __init__(self, df,
                 serial=False):
        self.df = standardize_data(df, method="log_and_diff").dropna().reset_index(drop=True)
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.usd_held = INITIAL_BALANCE
        self.eur_held = 0
        self.current_step = 0
        self.reward = 0
        self.serial = serial
        # trade history
        self.trades = []
        # our profit in last 5 trades
        self.returns = np.zeros((10))

        # index of episodes ( 1 episode equivalent to 1 week of trading)
        self.episode_indices = get_episode(self.df)
        self.action_space = spaces.Discrete(6)
        # observation space, includes: OLHC prices (normalized), close price (unnormalized),
        # time in minutes(encoded), day of week(encoded), action history, net worth changes history
        # both minutes, days feature are encoded using sin and cos function to retain circularity
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                            shape=(10, WINDOW_SIZE + 1),
                                            dtype=np.float16)
        self.metrics = Metric(INITIAL_BALANCE)
        self.last_trade_step = 0
        self.setup_active_df()
        self.agent_history = {"actions": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "net_worth": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "eur_held": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "usd_held": np.full(len(self.active_df), self.usd_held / BALANCE_NORM_FACTOR)}

    def get_metrics(self):
        return self.metrics

    def get_current_price(self):
        """
        :return: (float) closing price at time step x
        """
        return self.active_df.iloc[self.current_step + WINDOW_SIZE].Close

    def reset(self):
        """
        reset the environment to a fresh new state
        :return: (Gym.Box) new observation
        """
        self.reset_session()
        return self.next_observation()

    def setup_active_df(self):
        """
        select fragment of data we will use to train agent in this epoch
        :return: None
        """
        # if serial mode is enabled, we traverse through training data from 2012->2019
        # else we'll just jumping randomly betweek these times
        if self.serial:
            self.steps_left = len(self.df) - WINDOW_SIZE - 1
            self.frame_start = 0
        else:
            # pick random episode index from our db
            episode_index = np.random.randint(0, self.metrics.current_epoch * 8)
            # check if we have reached the end of dataset
            # and reroll the invalid index
            if episode_index >= len(self.episode_indices):
                episode_index = np.random.randint(0, len(self.episode_indices))

            (start_episode, end_episode) = self.episode_indices[episode_index]
            self.steps_left = end_episode - start_episode - WINDOW_SIZE
            self.frame_start = start_episode

        self.active_df = self.df[self.frame_start: self.frame_start +
                                 self.steps_left + WINDOW_SIZE + 1]

    def reset_variables(self):
        """
        reset all variables that involve with the environment
        :return: None
        """
        self.current_step = 0
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.usd_held = INITIAL_BALANCE
        self.eur_held = 0
        self.trades = []
        self.returns = np.zeros((10))
        self.agent_history = {"actions": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "net_worth": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "eur_held": np.zeros(len(self.active_df) + WINDOW_SIZE),
                              "usd_held": np.full(len(self.active_df), self.usd_held / BALANCE_NORM_FACTOR)}

    def reset_session(self):
        """
        # reset all variables and setup new training session
        :return: None
        """
        self.setup_active_df()
        self.reset_variables()

    def calculate_reward(self, action):
        """
        update reward we get at this time step
        :return: None
        """
        # calculate reward
        self.reward = 0
        profit = self.net_worth - self.prev_net_worth
        if profit > 0:
            self.reward += 0.2
        elif profit < 0:
            self.reward -= 0.2
        if np.mean(self.returns) > 0:
            self.reward += 0.4
        elif np.mean(self.returns) < 0:
            self.reward -= 0.4

        wining_trade_count = np.sum(self.returns > 0)
        losing_trade_count = np.sum(self.returns < 0)
        if wining_trade_count > 5:
            self.reward += wining_trade_count * 0.05
        if losing_trade_count > 5:
            self.reward -= losing_trade_count * 0.05

        if abs(self.eur_held) > LOT_SIZE * 2:
            self.reward -= 0.2 * abs(self.eur_held) / LOT_SIZE

    def step(self, action):
        """
        Perform choosen action and get the response from environment
        :param action: (int) 0 = hold, 1 = buy, 2 = sell, 3 = close, 4 = close and buy, 5 = close and sell
        :return: tuple contains (new observation, reward, isDone, {})
        """
        # perform action and update utility variables
        self.take_action(action, self.get_current_price())
        self.calculate_reward(action)

        # summary training process
        self.metrics.summary(action, self.net_worth, self.prev_net_worth, self.reward, self.eur_held)

        # get next observation and check whether we has finished this episode yet
        obs = self.next_observation()
        done = self.net_worth <= 0

        # reset session if we've reached the end of episode
        if self.steps_left == 0:
            self.reset_session()
            done = True

        return obs, self.reward, done, {}

    def take_action(self, action, sell_price):
        """
        Perform choosen action and then update our balance according to market state
        :param action: (int) 0 = hold, 1 = buy, 2 = sell, 3 = close, 4 = close and buy, 5 = close and sell
        :param sell_price: (float) current closing price
        :return: None
        """
        # in forex, we buy with current price + comission (it's normaly 3 pip
        # with eurusd pair)
        buy_price = sell_price + COMISSION
        # if action == BUY:
        #     action = SELL
        # elif action == SELL:
        #     action = BUY
        # elif action == CLOSE_AND_SELL:
        #     action = CLOSE_AND_BUY
        # elif action == CLOSE_AND_BUY:
        #     action = CLOSE_AND_SELL
        # elif action == HOLD:
        #     action = CLOSE
        # elif action == CLOSE:
        #     action = HOLD

        '''assume we have 100,000 usd and 0 eur
        assume current price is 1.5 (1 eur = 1.5 usd)
        assume comission = 3 pip = 0.0003
        => true buy price = 1.5003, sell price = 1.5
        buy 0.5 lot eur => we have 50,000 eur and (100,000 - 50,000 * 1.5003) = 24985 usd
        => out networth: 50,000 * 1.5 + 24985 = 99985 (we lose 3 pip, 1 pip = 5 usd,
        we are using 0.5 lot as defaut, if we buy 1 lot => 1 pip = 10 usd, correct!!! )'''
        if action == CLOSE_AND_BUY:  # buy eur
            self.close_and_buy(buy_price, sell_price)
        elif action == CLOSE_AND_SELL:  # sell eur
            self.close_and_sell(buy_price, sell_price)
        elif action == CLOSE:
            self.close_all_order(buy_price, sell_price)
        elif action == BUY:
            self.buy(buy_price)
        elif action == SELL:
            self.sell(sell_price)
        else:
            pass

        self.update(action)

    def update(self, action):
        sell_price = self.get_current_price()
        buy_price = sell_price + COMISSION
        # convert our networth to pure usd
        self.prev_net_worth = self.net_worth
        self.net_worth = self.usd_held + \
                         (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)

        self.update_history(sell_price, action)
        # increase training data after one epoch
        if self.metrics.num_step % 500000 == 0 and self.metrics.num_step > 0:
            self.metrics.current_epoch += 1
        self.steps_left -= 1
        self.current_step += 1
        self.returns[self.current_step % 10] = self.net_worth - self.prev_net_worth

    def update_history(self, sell_price, action):
        self.trades.append({'price': sell_price,
                            'eur_held': self.eur_held,
                            'usd_held': self.usd_held,
                            'net_worth': self.net_worth,
                            "prev_net_worth": self.prev_net_worth,
                            'type': ACTIONS[action]})

        # save these variables for training
        self.agent_history["actions"][self.current_step + WINDOW_SIZE + 1] = action
        self.agent_history["eur_held"][self.current_step + WINDOW_SIZE + 1] = self.eur_held / BALANCE_NORM_FACTOR
        self.agent_history["usd_held"][self.current_step + WINDOW_SIZE + 1] = self.usd_held / BALANCE_NORM_FACTOR
        self.agent_history["net_worth"][self.current_step + WINDOW_SIZE + 1] = \
            (self.net_worth - self.prev_net_worth) / LOT_SIZE

    def close_and_buy(self, buy_price, sell_price):
        self.usd_held += (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)
        # buy some eur
        self.eur_held = AMOUNT * LOT_SIZE
        self.usd_held -= AMOUNT * LOT_SIZE * buy_price

    def close_and_sell(self, buy_price, sell_price):
        self.usd_held += (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)
        # sell some eur
        self.eur_held = -AMOUNT * LOT_SIZE
        self.usd_held += AMOUNT * LOT_SIZE * sell_price

    def close_all_order(self, buy_price, sell_price):
        # close trade, release all eur we are holding (or buying)
        self.usd_held += (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)
        self.eur_held = 0

    def buy(self, buy_price):
        self.eur_held += AMOUNT * LOT_SIZE
        self.usd_held -= AMOUNT * LOT_SIZE * buy_price

    def sell(self, sell_price):
        self.eur_held -= AMOUNT * LOT_SIZE
        self.usd_held += AMOUNT * LOT_SIZE * sell_price

    def next_observation(self):
        # return the next observation of the environment
        end = self.current_step + WINDOW_SIZE + 1

        # atr = ta.average_true_range(self.active_df.High[self.current_step: end] * 100,
        #                             self.active_df.Low[self.current_step: end] * 100,
        #                             self.active_df.Close[self.current_step: end] * 100, n=9, fillna=True).to_numpy()
        # macd = ta.macd(self.active_df.Close[self.current_step: end] * 200, n_fast=9, n_slow=9, fillna=True).to_numpy()
        # rsi = ta.rsi(self.active_df.Close[self.current_step: end] / 100, fillna=True, n=9).to_numpy()

        obs = np.array([
            # self.active_df['Open'].values[self.current_step: end],
            # self.active_df['High'].values[self.current_step: end],
            # self.active_df['Low'].values[self.current_step: end],
            # self.active_df['NormedClose'].values[self.current_step: end],
            # self.active_df['Close'].values[self.current_step: end],
            self.active_df['CandleEmbededX'].values[self.current_step: end],
            self.active_df['CandleEmbededY'].values[self.current_step: end],
            self.active_df['TimeEncodedX'].values[self.current_step: end],
            self.active_df['TimeEncodedY'].values[self.current_step: end],
            self.active_df['DayEncodedX'].values[self.current_step: end],
            self.active_df['DayEncodedY'].values[self.current_step: end],
            # self.active_df['HighRiskTime'].values[self.current_step: end],
            self.agent_history["actions"][self.current_step: end],
            self.agent_history["net_worth"][self.current_step: end],
            self.agent_history["eur_held"][self.current_step: end],
            self.agent_history["usd_held"][self.current_step: end]
            # atr,
            # macd,
            # rsi
        ])

        return obs

    def render(self, mode='human'):
        """
        show information about trainin process
        :param mode: (string) if human mode is selected, display a additional
            graph that visualize trades, net worth, prices...
        :return: None
        """
        if mode == 'human':
            # init graph
            if not hasattr(self, 'visualization'):
                self.visualization = StockTradingGraph(
                    self.df, "Reward visualization")
            # render it if we have enough information
            if self.current_step > WINDOW_SIZE and mode == 'human':
                self.visualization.render(
                    self.current_step,
                    self.net_worth,
                    self.reward,
                    window_size=WINDOW_SIZE)
        # print out some statistic about agent
        if self.metrics.num_step % 50 == 0:
            # save these variables for plotting
            self.metrics.update_for_plotting()
            print("{:<25s}{:>5.2f}".format("current step:", self.current_step))
            print("{:<25s}{:>5.2f}".format("Total win trades:", self.metrics.win_trades))
            print("{:<25s}{:>5.2f}".format("Total lose trades:", self.metrics.lose_trades))
            print("{:<25s}{:>5.2f}".format("Total hold trades:", self.metrics.hold_trades))
            print("{:<25s}{:>5.2f}".format("Avg win value:", self.metrics.avg_win_value))
            print("{:<25s}{:>5.2f}".format("Avg lose value:", self.metrics.avg_lose_value))
            print("{:<25s}{:>5.2f}".format("Avg reward:", self.metrics.avg_reward / self.metrics.num_step))
            print("{:<25s}{:>5.2f}".format("Highest net worth:", self.metrics.highest_net_worth))
            print("{:<25s}{:>5.2f}".format("Lowest net worth:", self.metrics.lowest_net_worth))
            print("{:<25s}{:>5.2f}".format("Most profit trade win:", self.metrics.most_profit_trade))
            print("{:<25s}{:>5.2f}".format("Worst trade lose:", self.metrics.worst_trade))
            print("{:<25s}{:>5.2f}".format("Win ratio:", self.metrics.win_trades / (self.metrics.lose_trades + 1 + self.metrics.win_trades)))
            print('-' * 80)
