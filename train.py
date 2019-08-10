import argparse
import pandas as pd

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from util import plot_metrics, evaluate_train_set, evaluate_test_set
from env import TradingEnv, LSTM_Env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        default="train",
                        help="specific model to run our model, available options are: train, test")
    parser.add_argument("--test_mode", type=str,
                        default="single",
                        help="specific model to run our model, available options are: train, test")
    args = parser.parse_args()

    # read data and init environments
    train_df = pd.read_csv('./data/EURUSD_m15_train.csv', index_col=0)
    test_df = pd.read_csv('./data/EURUSD_m15_test.csv', index_col=0)
    # The algorithms require a vectorized environment to run
    train_env = DummyVecEnv([lambda: TradingEnv(train_df, serial=True)])
    test_env = DummyVecEnv([lambda: TradingEnv(test_df, serial=True)])

    if args.mode == "train":
        print("Training started")
        model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log='./logs', nminibatches=1)
        model.learn(total_timesteps=1000000, seed=69)
        model.save("./models/mlp_model")
        print("Training's done, saved model to ./models/mlp_model")
    else:
        # load pre-trained model
        print("Loading model at: ./models/mlp_model")
        model = PPO2.load("./models/mlp_model.pkl")

        print("Start testing on train set")
        evaluate_train_set(model, train_env)

        if args.test_mode == 'double':
            print("Start testing on test set")
            evaluate_test_set(model, test_env)

        print("Testing's comeplete")




