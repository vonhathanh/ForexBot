import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env import TradingEnv

import pandas as pd

df = pd.read_csv('./data/EURUSD_m15.csv', index_col=0)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: TradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50)

obs = env.reset()
for i in range(len(df['Time'])):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()