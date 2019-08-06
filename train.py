import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from util import plot_metrics

from env import TradingEnv

import pandas as pd

train_df = pd.read_csv('./data/EURUSD_m15_train.csv', index_col=0)
test_df = pd.read_csv('./data/EURUSD_m15_test.csv', index_col=0)

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
test_env = DummyVecEnv([lambda: TradingEnv(test_df, serial=True)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log='./logs')
model.learn(total_timesteps=10000, seed=69)
model.save("./models/mlp_model")

# model = PPO2.load("./models/mlp_model.pkl")

# print("Start testing on test set")
# obs = test_env.reset()
# for i in range(len(test_df)):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = test_env.step(action)
#     test_env.render(mode='verbose')
#     if done:
#         test_env.reset()
#
# plot_metrics(test_env.get_attr('metrics')[0])

# print("Start testing on train set (for overfitting check")
#
# obs = train_env.reset()
# for i in range(3000):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = train_env.step(action)
#     train_env.render(mode='verbose')
#     if done:
#         train_env.reset()
#
# plot_metrics(train_env.get_attr('metrics')[0])