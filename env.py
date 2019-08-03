import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from trading_graph import StockTradingGraph

from gym import spaces


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_TRADING_SESSION = 2000
    LOT_SIZE = 100000

    def __init__(self, df,
                 look_back_window_size=50,
                 commission=0.0003,
                 initial_balance=1000*1000,
                 serial=False):

        super(TradingEnv, self).__init__()
        self.df = df.dropna().reset_index()
        self.look_back_window_size = look_back_window_size
        self.initial_balance = initial_balance
        self.net_worth = self.initial_balance
        self.eur_held = 0
        self.usd_held = self.initial_balance
        self.trades = []
        self.current_step = 0
        self.reward = 0
        self.account_history = np.repeat([[self.net_worth], [0], [self.net_worth], [0]],
                                         self.look_back_window_size + 1,
                                         axis=1)
        self.commission = commission
        self.serial = serial
        self.scaler = StandardScaler()
        self.visualization = StockTradingGraph(
            self.df, "Reward visualization")

        # TODO: do we need to add buy stop, sell stop, buy limit, sell limit to action space? (may be not, start simple first)
        # action: buy, sell, hold <=> 0, 1, 2
        # amount: 0.1, 0.2, 0.5, 1, 2, 5 lot (ignore amount for now, we will use 0.5 lot as default
        # => 3 actions available
        self.action_space = spaces.Discrete(3)
        # observe the OHCL values, networth, and trade history (eur held, usd held, actions)
        # TODO: consider add time to observation space
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                             shape=(8, look_back_window_size + 1),
                                            dtype=np.float16)

        # these properties are our metric for comparing different models
        self.win_trades = 0
        self.lose_trades = 0
        self.avg_reward = 0
        self.num_step = 0
        self.most_profit_trade = 0
        self.worst_trade = 0
        self.highest_networth = self.initial_balance
        self.lowest_networth = self.initial_balance

    def reset(self):
        self.net_worth = self.initial_balance
        self.eur_held = 0
        self.usd_held = self.initial_balance

        self.reset_session()

        self.account_history = np.repeat([[self.net_worth], [0], [self.net_worth], [0]],
                                         self.look_back_window_size + 1,
                                         axis=1)
        self.trades = []

        return self.next_observation()

    def reset_session(self):
        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.look_back_window_size - 1
            self.frame_start = self.look_back_window_size
        else:
            self.steps_left = np.random.randint(200, self.MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.look_back_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.look_back_window_size : self.frame_start + self.steps_left]

    def next_observation(self):
        end = self.current_step + self.look_back_window_size + 1

        obs = np.array([
            self.active_df['Open'].values[self.current_step: end],
            self.active_df['High'].values[self.current_step: end],
            self.active_df['Low'].values[self.current_step: end],
            self.active_df['Close'].values[self.current_step: end],
        ])

        scaled_history = self.scaler.fit_transform(self.account_history)
        obs = np.append(obs, scaled_history[:, -(self.look_back_window_size + 1) :], axis=0)
        return obs

    def get_current_price(self):
        return self.active_df.iloc[self.current_step].Close

    def step(self, action):
        current_price = self.get_current_price()
        self.take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.usd_held = 0
            self.eur_held = 0
            self.reset_session()

        obs = self.next_observation()
        reward = self.net_worth - self.prev_networth
        done = self.net_worth <= 0

        self.reward = reward

        '''
                We define some metrics to measure model's performance in test environment
                - Number of win trades, lose trades
                - Average reward
                - Maximum networth achieved
                - Minium networth achieved
                - biggest win, lose trade
                '''
        if self.net_worth <= self.prev_networth:
            self.lose_trades += 1
        else:
            self.win_trades += 1
        self.num_step += 1
        self.avg_reward = self.avg_reward + 1 / self.num_step * (self.reward - self.avg_reward)

        if self.net_worth > self.highest_networth:
            self.highest_networth = self.net_worth
        if self.net_worth < self.lowest_networth:
            self.lowest_networth = self.net_worth

        profit = self.net_worth - self.prev_networth
        if profit > self.most_profit_trade:
            self.most_profit_trade = profit
        if profit < self.worst_trade:
            self.worst_trade = profit

        return obs, reward, done, {}

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
            self.eur_held += amount * self.LOT_SIZE
            self.usd_held -= amount * self.LOT_SIZE * buy_price

        elif action == 2:  # sell eur => decrease eur held, increase usd held
            self.eur_held -= amount * self.LOT_SIZE
            self.usd_held += amount * self.LOT_SIZE * sell_price
        else:  # hold
            pass

        self.prev_networth = self.net_worth
        # convert our networth to pure usd
        self.net_worth = self.usd_held + (self.eur_held * sell_price if self.eur_held > 0 else self.eur_held * buy_price)

        if action == 1 or action == 2:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': amount,
                                'type': "buy" if action == 1 else "sell"})

        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [self.eur_held],
            [self.usd_held],
            [action],
        ], axis=1)


    def render(self, mode='human'):
        if self.current_step > self.look_back_window_size and mode == 'human':
            self.visualization.render(
                self.current_step, self.net_worth, self.reward, window_size=self.look_back_window_size)

        if self.current_step % 50 == 0:
            print("{:<30s}{:>5.2f}".format("current step:", self.current_step))
            print("{:<30s}{:>5.2f}".format("Total win trades:", self.win_trades))
            print("{:<30s}{:>5.2f}".format("Total lose trades:", self.lose_trades))
            print("{:<30s}{:>5.2f}".format("Avg reward:", self.avg_reward))
            print("{:<30s}{:>5.2f}".format("Highest net worth:", self.highest_networth))
            print("{:<30s}{:>5.2f}".format("Lowest net worth:", self.lowest_networth))
            print("{:<30s}{:>5.2f}".format("Most profit trade win:", self.most_profit_trade))
            print("{:<30s}{:>5.2f}".format("Worst trade lose:", self.worst_trade))







