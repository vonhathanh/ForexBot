import gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from trading_graph import StockTradingGraph
from gym import spaces
from util import standardize_data, get_episode
from metrics import Metric

MAX_TRADING_SESSION = 2000
LOT_SIZE = 100000
WINDOW_SIZE = 8
INITIAL_BALANCE = 100000
COMISSION = 0.0001
ACTION = {0: "hold", 1: "buy", 2: "sell", 3: "close"}


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, df,
                 serial=False):

        super(TradingEnv, self).__init__()
        self.df = standardize_data(df, method="log_and_diff").dropna().reset_index(drop=True)
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.usd_held = INITIAL_BALANCE
        self.eur_held = 0
        self.current_step = 0
        self.reward = 0
        self.serial = serial
        self.trades = []
        self.returns = np.array([0, 0, 0, 0, 0])

        # TODO: do we need to add buy stop, sell stop, buy limit, sell limit to action space? (may be not, start simple first)
        # action: buy, sell, hold <=> 0, 1, 2
        # amount: 0.1, 0.2, 0.5, 1, 2, 5 lot (ignore amount for now, we will use 0.5 lot as default
        # => 3 actions available
        self.action_space = spaces.Discrete(3)
        # observe the OHCL values, networth, time, and trade history (eur held, usd held, actions)
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                             shape=(4, WINDOW_SIZE + 1),
                                            dtype=np.float16)
        self.metrics = Metric(INITIAL_BALANCE)
        np.random.seed(69)

    def get_metrics(self):
        return self.metrics

    def reset(self):
        # reset the environment and return fresh new state
        self.reset_session()
        return self.next_observation()

    def reset_variables(self):
        # reset all variables that involve with the environment
        self.current_step = 0
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.usd_held = INITIAL_BALANCE
        self.eur_held = 0
        self.trades = []
        self.returns = np.array([0, 0, 0, 0, 0])

    def setup_active_df(self):
        if self.serial:
            self.steps_left = len(self.df) - WINDOW_SIZE - 1
            self.frame_start = WINDOW_SIZE
        else:
            self.steps_left = np.random.randint(500, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(WINDOW_SIZE, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - WINDOW_SIZE : self.frame_start + self.steps_left]

    def reset_session(self):
        # reset all variables that involve with the environment
        self.reset_variables()
        self.setup_active_df()

    def next_observation(self):
        # return the next observation of the environment
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
        return self.active_df.iloc[self.current_step].Close

    def step(self, action):
        self.take_action(action, self.get_current_price())
        self.steps_left -= 1
        self.current_step += 1
        self.returns[self.current_step % 5] = self.net_worth - self.prev_net_worth

        if self.prev_net_worth <= 0 or self.net_worth <= 0:
            self.reward = -5
        else:
            self.reward = np.log(self.net_worth / (self.prev_net_worth + 0.0001))

        if self.returns.mean() > 0:
            self.reward += 0.5
        elif self.returns.mean() < 0:
            self.reward -= 1.0

        obs = self.next_observation()
        done = self.net_worth <= 0

        self.metrics.summary(action, self.net_worth, self.prev_net_worth, self.reward)

        if self.steps_left == 0:
            self.reset_session()
            done = True

        return obs, self.reward, done, {}

    def take_action(self, action, current_price):
        amount = 0.5
        # in forex, we buy with current price + comission (it's normaly 3 pip with eurusd pair)
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
        # elif action == 3:
        #     # close trade, release all eur we are holding (or buying)
        #     self.usd_held += (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)
        #     self.eur_held = 0
        else:
            pass

        self.prev_net_worth = self.net_worth
        # convert our networth to pure usd
        self.net_worth = self.usd_held + (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)

        self.trades.append({'price': current_price,
                            'eur_held': self.eur_held,
                            'usd_held': self.usd_held,
                            'net_worth': self.net_worth,
                            'type': ACTION[action]})

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'visualization'):
                self.visualization = StockTradingGraph(self.df, "Reward visualization")
            if self.current_step > WINDOW_SIZE and mode == 'human':
                self.visualization.render(
                    self.current_step, self.net_worth, self.reward, window_size=WINDOW_SIZE)

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
            print("{:<25s}{:>5.2f}".format("Win ratio:", self.metrics.win_trades /
                                           (self.metrics.lose_trades + 1 + self.metrics.win_trades)))
            print('-'*80)


class LSTM_Env(TradingEnv):

    def __init__(self, df,
                 serial=False):
        super().__init__(df, serial)

        self.episode_indices = get_episode(self.df)

        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                             shape=(11, WINDOW_SIZE + 1),
                                            dtype=np.float16)
        self.setup_active_df()
        self.actions = np.zeros(len(self.active_df) + WINDOW_SIZE)
        self.net_worth_history = np.zeros(len(self.active_df) + WINDOW_SIZE)

    def setup_active_df(self):
        if self.serial:
            self.steps_left = len(self.df) - WINDOW_SIZE - 1
            self.frame_start = 0
        else:
            # pick random episode index from our db
            episode_index = np.random.randint(0, len(self.episode_indices))
            (start_episode, end_episode) = self.episode_indices[episode_index]

            self.steps_left = end_episode - start_episode - WINDOW_SIZE
            self.frame_start = start_episode

        self.active_df = self.df[self.frame_start: self.frame_start + self.steps_left + WINDOW_SIZE + 1]

    def reset_variables(self):
        super().reset_variables()
        self.actions = np.zeros(len(self.active_df) + WINDOW_SIZE + 1)
        self.net_worth_history = np.zeros(len(self.active_df) + WINDOW_SIZE + 1)

    def reset_session(self):
        self.setup_active_df()
        self.reset_variables()

    def take_action(self, action, current_price):
        super().take_action(action, current_price)
        self.actions[self.current_step + WINDOW_SIZE] = action
        self.net_worth_history[self.current_step + WINDOW_SIZE] = (self.net_worth - self.prev_net_worth) / LOT_SIZE

    def next_observation(self):
        # return the next observation of the environment
        end = self.current_step + WINDOW_SIZE + 1
        obs = np.array([
            self.active_df['Open'].values[self.current_step: end],
            self.active_df['High'].values[self.current_step: end],
            self.active_df['Low'].values[self.current_step: end],
            self.active_df['NormedClose'].values[self.current_step: end],
            self.active_df['Close'].values[self.current_step: end],
            self.active_df['timeEncodedX'].values[self.current_step: end],
            self.active_df['timeEncodedY'].values[self.current_step: end],
            self.active_df['dayEncodedX'].values[self.current_step: end],
            self.active_df['dayEncodedY'].values[self.current_step: end],
            self.actions[self.current_step: end],
            self.net_worth_history[self.current_step: end]
        ])

        return obs



