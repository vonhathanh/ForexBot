import argparse
import pandas as pd
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from util import evaluate_train_set, evaluate_test_set, SRC_DIR
from custom_policy import CustomLSTMPolicy
from env import LSTM_Env


def make_env(seed, df, serial):
    def _init():
        env = LSTM_Env(df, serial)
        env.seed(seed)
        return env
    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        default="train",
                        help="specific mode to run our model, available options are: train, test")
    parser.add_argument("--test_mode", type=str,
                        default="single",
                        help="specific mode to run test, available options are: single, double, "
                             "single means test with trained dataset. Double means test with train and test data set")
    parser.add_argument("--model", type=str,
                        default="lstm",
                        help="specific model to run, available models are: mlp, lstm")
    parser.add_argument("--render", type=str,
                        default="verbose",
                        help="specific display mode, available models are: verbose, human")
    args = parser.parse_args()

    # read data and init environments
    train_df_m15 = pd.read_csv(os.path.join(SRC_DIR, "..", "data", "EURUSD_m15_train.csv"), index_col=0)
    train_df_h1 = pd.read_csv(os.path.join(SRC_DIR, "..", "data", "EURUSD_h1_train.csv"), index_col=0)
    test_df_m15 = pd.read_csv(os.path.join(SRC_DIR, "..", "data", "EURUSD_m15_test.csv"), index_col=0)
    test_df_h1 = pd.read_csv(os.path.join(SRC_DIR, "..", "data", "EURUSD_h1_test.csv"), index_col=0)
    # The algorithms require a vectorized environment to run
    serial = False
    if args.mode == "test":
        serial = True

    if args.model == 'mlp':
        train_env = DummyVecEnv([lambda: LSTM_Env(train_df_m15, train_df_h1, serial)])
        test_env = DummyVecEnv([lambda: LSTM_Env(test_df_m15, test_df_h1, serial)])
        model = PPO2(MlpPolicy, train_env, gamma=0.95, verbose=1, tensorboard_log=os.path.join(SRC_DIR, "..", "logs"))
    else:
        train_env = DummyVecEnv([lambda: LSTM_Env(train_df_m15, train_df_h1, serial)])
        test_env = DummyVecEnv([lambda: LSTM_Env(test_df_m15, train_df_h1, serial)])

        model = PPO2(CustomLSTMPolicy,
                     train_env,
                     gamma=0.95,
                     verbose=1,
                     tensorboard_log=os.path.join(SRC_DIR, "..", "logs"),
                     nminibatches=1,
                     n_steps=16)

    save_path = os.path.join(SRC_DIR, "..", "models", args.model + "_model")

    render_mode = args.render

    if args.mode == "train":
        print("Training started")
        model.learn(total_timesteps=100000, seed=69)
        model.save(save_path)
        print("Training's done, saved model to: ", save_path)
    else:
        # load pre-trained model
        print("Loading model at: ", save_path)
        model = PPO2.load(save_path)
        print("Start testing on train set")
        evaluate_train_set(model, train_env, 1000, render_mode)

        if args.test_mode == 'double':
            print("Start testing on test set")
            evaluate_test_set(model, test_env, 1000, render_mode)

        print("Testing's comeplete")






