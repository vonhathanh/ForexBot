import gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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
# look back window, agent will use this to know what happended in the past 8 time frame
WINDOW_SIZE = 8
# comission when we making a trade
COMISSION = 0.0002

INITIAL_BALANCE = 100000
ACTIONS = {0: "hold", 1: "buy", 2: "sell", 3: "close"}


class TradingEnv(gym.Env):

    def __init__(self, df, serial=False):
        """
        :param df: (Pandas DataFrame) The current Dataframe we interested in
        :param serial: (bool) Decide whether we want to split the data to multiple part or not
        """
        super(TradingEnv, self).__init__()
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
        self.returns = np.array([0, 0, 0, 0, 0])

        # TODO: do we need to add buy stop, sell stop, buy limit, sell limit to action space? (may be not, start simple first)
        # action: buy, sell, hold, close <=> 0, 1, 2, 3
        # amount: 0.1, 0.2, 0.5, 1, 2, 5 lot (ignore amount for now, we will use 0.5 lot as default
        # => 4 actions available
        self.action_space = spaces.Discrete(4)
        # observe the OHCL values, networth, time, and trade history (eur held,
        # usd held, actions)
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                            shape=(4, WINDOW_SIZE + 1),
                                            dtype=np.float16)
        self.metrics = Metric(INITIAL_BALANCE)
        np.random.seed(69)

    def get_metrics(self):
        return self.metrics

    def reset(self):
        """
        reset the environment to a fresh new state
        :return: (Gym.Box) new observation
        """
        self.reset_session()
        return self.next_observation()

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
        self.returns = np.array([0, 0, 0, 0, 0])

    def setup_active_df(self):
        """
        Determine which part of data frame will be used for training or testing
        :return: None
        """
        if self.serial:
            self.steps_left = len(self.df) - WINDOW_SIZE - 1
            self.frame_start = WINDOW_SIZE
        else:
            self.steps_left = np.random.randint(500, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                WINDOW_SIZE, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - \
            WINDOW_SIZE: self.frame_start + self.steps_left]

    def reset_session(self):
        """
        # reset all variables and setup new training session
        :return: None
        """
        self.reset_variables()
        self.setup_active_df()

    def next_observation(self):
        """
        :return: (Gym.Box) the next observation of the environment
        """
        end = self.current_step + WINDOW_SIZE + 1

        obs = np.array([
            # self.active_df['NormalizedTime'].values[self.current_step: end],
            self.active_df['Open'].values[self.current_step: end],
            self.active_df['High'].values[self.current_step: end],
            self.active_df['Low'].values[self.current_step: end],
            self.active_df['NormedClose'].values[self.current_step: end],
        ])

        return obs

    def get_current_price(self):
        """
        :return: (float) closing price at time step x
        """
        return self.active_df.iloc[self.current_step].Close

    def step(self, action):
        """
        Perform choosen action and get the response from environment
        :param action: (int) 0 = hold, 1 = buy, 2 = sell
        :return: tuple contains (new observation, reward, isDone, {})
        """
        # perform action and update utility variables
        self.take_action(action, self.get_current_price())
        self.steps_left -= 1
        self.current_step += 1
        self.returns[self.current_step %
                     5] = self.net_worth - self.prev_net_worth

        # calculate reward
        if self.prev_net_worth <= 0 or self.net_worth <= 0:
            self.reward = -5
        else:
            if self.net_worth > self.prev_net_worth:
                self.reward = 1
            else:
                self.reward = -0.5
            if np.sum(self.returns >= 0) >= 3:
                self.reward += 1
            elif np.sum(self.returns < 0) >= 3:
                self.reward -= 0.5

        # get next observation and check whether we has finished this episode yet
        obs = self.next_observation()
        done = self.net_worth <= 0

        # summary training process
        self.metrics.summary(
            action,
            self.net_worth,
            self.prev_net_worth,
            self.reward)

        # reset session if we've reached the end of episode
        if self.steps_left == 0:
            self.reset_session()
            done = True

        return obs, self.reward, done, {}

    def take_action(self, action, current_price):
        """
        Perform choosen action and then update our balance according to market state
        :param action: (int) 0 = hold, 1 = buy, 2 = sell
        :param current_price: (float) current closing price
        :return: None
        """
        amount = 0.5
        # in forex, we buy with current price + comission (it's normaly 3 pip
        # with eurusd pair)
        buy_price = current_price + COMISSION
        sell_price = current_price

        '''assume we have 100,000 usd and 0 eur
        assume current price is 1.5 (1 eur = 1.5 usd)
        assume comission = 3 pip = 0.0003
        => true buy price = 1.5003, sell price = 1.5
        buy 0.5 lot eur => we have 50,000 eur and (100,000 - 50,000 * 1.5003) = 24985 usd
        => out networth: 50,000 * 1.5 + 24985 = 99985 (we lose 3 pip, 1 pip = 5 usd,
        we are using 0.5 lot as defaut, if we buy 1 lot => 1 pip = 10 usd, correct!!! )'''
        if action == 1:  # buy eur, sell usd => increase eur held, decrease usd held
            self.eur_held += amount * LOT_SIZE
            self.usd_held -= amount * LOT_SIZE * buy_price

        elif action == 2:  # sell eur => decrease eur held, increase usd held
            self.eur_held -= amount * LOT_SIZE
            self.usd_held += amount * LOT_SIZE * sell_price
        elif action == 3:
            # close trade, release all eur we are holding (or buying)
            self.usd_held += (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)
            self.eur_held = 0
        else:
            pass

        self.prev_net_worth = self.net_worth
        # convert our networth to pure usd
        self.net_worth = self.usd_held + \
            (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)

        self.trades.append({'price': current_price,
                            'eur_held': self.eur_held,
                            'usd_held': self.usd_held,
                            'net_worth': self.net_worth,
                            'type': ACTIONS[action]})

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


# env that use lstm architecture to train the model
class LSTM_Env(TradingEnv):

    def __init__(self, df,
                 serial=False):
        super().__init__(df, serial)
        # epoch counter, for each epoch passed (about 100k steps),
        # we will increase the epoch and add 8 more weeks to training data
        self.current_epoch = 1
        # index of current episode ( 1 episode equivalent to 1 week of trading)
        self.episode_indices = get_episode(self.df)
        # observation space, includes: OLHC prices (normalized), close price (unnormalized),
        # time in minutes(encoded), day of week(encoded), action history, net worth changes history
        # both minutes, days feature are encoded using sin and cos function to retain circularity
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                            shape=(12, WINDOW_SIZE + 1),
                                            dtype=np.float16)
        self.setup_active_df()
        self.actions = np.zeros(len(self.active_df) + WINDOW_SIZE)
        self.net_worth_history = np.zeros(len(self.active_df) + WINDOW_SIZE)

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
            episode_index = np.random.randint(0, self.current_epoch * 8)
            (start_episode, end_episode) = self.episode_indices[episode_index]
            self.steps_left = end_episode - start_episode - WINDOW_SIZE
            self.frame_start = start_episode

        self.active_df = self.df[self.frame_start: self.frame_start +
                                 self.steps_left + WINDOW_SIZE + 1]

    def reset_variables(self):
        super().reset_variables()
        self.actions = np.zeros(len(self.active_df) + WINDOW_SIZE + 1)
        self.net_worth_history = np.zeros(
            len(self.active_df) + WINDOW_SIZE + 1)

    def reset_session(self):
        self.setup_active_df()
        self.reset_variables()

    def take_action(self, action, current_price):
        super().take_action(action, current_price)
        # save these variables for training
        self.actions[self.current_step + WINDOW_SIZE] = action
        self.net_worth_history[self.current_step + WINDOW_SIZE] = (
            self.net_worth - self.prev_net_worth) / LOT_SIZE
        # increase training data after one epoch
        if self.metrics.num_step % 100000 == 0 and self.metrics.num_step > 0:
            self.current_epoch += 1

    def next_observation(self):
        # return the next observation of the environment
        end = self.current_step + WINDOW_SIZE + 1
        obs = np.array([
            self.active_df['Open'].values[self.current_step: end],
            self.active_df['High'].values[self.current_step: end],
            self.active_df['Low'].values[self.current_step: end],
            self.active_df['NormedClose'].values[self.current_step: end],
            self.active_df['Close'].values[self.current_step: end],
            self.active_df['TimeEncodedX'].values[self.current_step: end],
            self.active_df['TimeEncodedY'].values[self.current_step: end],
            self.active_df['DayEncodedX'].values[self.current_step: end],
            self.active_df['DayEncodedY'].values[self.current_step: end],
            self.active_df['HighRiskTime'].values[self.current_step: end],
            self.actions[self.current_step: end],
            self.net_worth_history[self.current_step: end]
        ])

        return obs
