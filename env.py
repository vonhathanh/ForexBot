import gym
import numpy as np

from sklearn.preprocessing import StandardScaler
from trading_graph import StockTradingGraph
from gym import spaces
from util import standardize_data
from metrics import Metric

MAX_TRADING_SESSION = 2000
LOT_SIZE = 100000

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, df,
                 look_back_window_size=50,
                 commission=0.0001,
                 initial_balance=100*1000,
                 serial=False,
                 random=False):

        super(TradingEnv, self).__init__()
        self.df = standardize_data(df).dropna().reset_index(drop=True)
        self.look_back_window_size = look_back_window_size
        self.initial_balance = initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.net_worth
        self.eur_held = 0
        self.usd_held = self.initial_balance
        self.trades = []
        self.current_step = 0
        self.reward = 0
        self.random = random
        self.commission = commission
        self.serial = serial

        # TODO: do we need to add buy stop, sell stop, buy limit, sell limit to action space? (may be not, start simple first)
        # action: buy, sell, hold <=> 0, 1, 2
        # amount: 0.1, 0.2, 0.5, 1, 2, 5 lot (ignore amount for now, we will use 0.5 lot as default
        # => 3 actions available
        self.action_space = spaces.Discrete(3)
        # observe the OHCL values, networth, time, and trade history (eur held, usd held, actions)
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                             shape=(4, look_back_window_size + 1),
                                            dtype=np.float16)
        self.metrics = Metric(initial_balance)
        np.random.seed(69)

    def get_metrics(self):
        return self.metrics

    def reset(self):
        # reset the environment and return fresh new state
        self.reset_session()
        return self.next_observation()

    def reset_session(self):
        # reset all variables that involve with the environment
        self.current_step = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.eur_held = 0
        self.usd_held = self.initial_balance
        self.trades = []

        if self.serial:
            self.steps_left = len(self.df) - self.look_back_window_size - 1
            self.frame_start = self.look_back_window_size
        else:
            self.steps_left = np.random.randint(500, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.look_back_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.look_back_window_size : self.frame_start + self.steps_left]

    def next_observation(self):
        # return the next observation of the environment
        end = self.current_step + self.look_back_window_size + 1

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
        current_price = self.get_current_price()
        if self.random:
            action = np.random.randint(3)
        self.take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1
        if self.prev_net_worth <= 0 or self.net_worth <= 0:
            self.reward = -5
        else:
            self.reward = np.log(self.net_worth / (self.prev_net_worth + 0.0001))

        obs = self.next_observation()
        done = self.net_worth <= 0

        self.metrics.summary(action, self.net_worth, self.prev_net_worth, self.reward)

        if self.steps_left == 1:
            self.reset_session()
            done = True

        return obs, self.reward, done, {}

    def take_action(self, action, current_price):
        amount = 0.5
        # in forex, we buy with current price + comission (it's normaly 3 pip with eurusd pair)
        buy_price = current_price + self.commission
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
        else:  # hold
            pass

        self.prev_net_worth = self.net_worth
        # convert our networth to pure usd
        self.net_worth = self.usd_held + (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)

        if action == 1 or action == 2:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': amount,
                                'type': "buy" if action == 1 else "sell"})

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'visualization'):
                self.visualization = StockTradingGraph(self.df, "Reward visualization")
            if self.current_step > self.look_back_window_size and mode == 'human':
                self.visualization.render(
                    self.current_step, self.net_worth, self.reward, window_size=self.look_back_window_size)

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

    def __init__(self, df, look_back_window_size=50,
                 commission=0.0003,
                 initial_balance=100*1000,
                 serial=False,
                 random=False):
        super().__init__(df, look_back_window_size,
                 commission,
                 initial_balance,
                 serial,
                 random)

        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                             shape=(9, ),
                                            dtype=np.float16)

    def next_observation(self):
        obs = np.array([
            # self.active_df['NormalizedTime'].values[self.current_step: end],
            self.active_df['Open'].values[self.current_step],
            self.active_df['High'].values[self.current_step],
            self.active_df['Low'].values[self.current_step],
            self.active_df['NormedClose'].values[self.current_step],
            self.active_df['NormalizedTime'].values[self.current_step],
            self.net_worth / LOT_SIZE,
            self.prev_net_worth / LOT_SIZE,
            self.eur_held / LOT_SIZE,
            self.usd_held / LOT_SIZE
        ])

        return obs

