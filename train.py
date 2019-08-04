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
test_env = DummyVecEnv([lambda: TradingEnv(test_df)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log='./logs')
model.learn(total_timesteps=1000)
model.save("./models/mlp_model.model")

obs = test_env.reset()
for i in range(len(test_df)):
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render(mode='human')
    if done:
        test_env.reset()
        print("oh yeah, trading done")

plot_metrics(test_env.get_attr('metrics')[0])
