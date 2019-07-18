import gym
import numpy as np

from gym import spaces


class TradingAgent(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_TRADING_SESSION = 100000
    LOT_SIZE = 100000

    def __init__(self, df,
                 look_back_window_size=50,
                 commission=0.001,
                 initial_balance=1000*100,
                 serial=False):

        super(TradingAgent, self).__init__()
        self.df = df.dropna().reset_index()
        self.look_back_window_size = look_back_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial

        # TODO: do we need to add buy stop, sell stop, buy limit,
        #  sell limit to action space? (may be not, start simple first)
        # action: buy, sell, hold
        # amount: 0.1, 0.2, 0.5, 1, 2, 5 lot
        # => 3x6 actions available
        self.action_space = spaces.MultiDiscrete([3, 6])
        # observe the OHCL values, networth, and trade history
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(10, look_back_window_size + 1),
                                            dtype=np.float16)

    def reset(self):
        self.networth = self.initial_balance
        self.eur_held = 0
        self.usd_held = self.initial_balance

        self.reset_session()

        self.account_history = np.repeat([self.networth, [0], [0], [0], [0]],
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
            self.steps_left = np.random.randint(1, self.MAX_TRADING_SESSION)
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

    def step(self, action):
        current_price = self.get_current_price() + self.commission
        self.take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.usd_held = 0
            self.eur_held = 0
            self.reset_session()

        obs = self.next_observation()
        reward = self.networth
        done = self.networth <= 0

        return obs, reward, done, {}

    def take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1]

        buy_price = current_price + self.commission
        sell_price = current_price

        if action_type == 1:  # buy
            self.eur_held += amount * self.LOT_SIZE
            self.usd_held -= self.eur_held * buy_price

        elif action_type == 2:  # sell
            self.eur_held -= amount * self.LOT_SIZE
            self.usd_held += self.eur_held * sell_price
        else:  # hold
            pass

        self.networth = self.usd_held + self.eur_held * sell_price

        if action_type == 1 or action_type == 2:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': amount,
                                'type': "buy" if action_type == 1 else "sell"})

        self.account_history = np.append(self.account_history, [
            self.networth,
            self.eur_held,
            self.usd_held,
        ], axis=1)







