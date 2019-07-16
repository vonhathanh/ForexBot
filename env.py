import gym
from gym import spaces

class TradingAgent(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df,
                 lookback_window_size=50,
                 commission=10,
                 initial_balance=10000,
                 serial=False):

        super(TradingAgent, self).__init__()
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial

        # action of the format buy 1/10, 2/10,..., 10/10
        self.action_space = spaces.MultiDiscrete([3, 10])
        # observe the OHCLV values, networth, and trade history
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(10, lookback_window_size + 1),
                                            dtype=np.float16)

    def reset(self):
        self.balance = self.initial_balance
        self.networth = self.initial_balance
        self.eurusd_held = 0.0

        self.reset_session()

        self.account_history = np.repeat([self.networth, [0], [0], [0], [0]],
                                         self.lookback_window_size + 1,
                                         axis=1)

        self.trades = []

        return self.next_observation()

