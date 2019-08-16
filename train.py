import argparse
import pandas as pd

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from util import evaluate_train_set, evaluate_test_set
from env import TradingEnv, LSTM_Env


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
    args = parser.parse_args()

    # read data and init environments
    train_df = pd.read_csv('./data/EURUSD_m15_train.csv', index_col=0)
    test_df = pd.read_csv('./data/EURUSD_m15_test.csv', index_col=0)
    # The algorithms require a vectorized environment to run

    if args.model == 'mlp':
        train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
        test_env = DummyVecEnv([lambda: TradingEnv(test_df, serial=True)])
        model = PPO2(MlpPolicy, train_env, gamma=0.95, verbose=1, tensorboard_log='./logs')
    else:
        train_env = DummyVecEnv([lambda: LSTM_Env(train_df)])
        test_env = DummyVecEnv([lambda: LSTM_Env(test_df, serial=True)])

        policy_kwargs = dict(net_arch=[64, 64, 'lstm'])
        model = PPO2(MlpLstmPolicy,
                     train_env,
                     gamma=0.98,
                     verbose=1,
                     tensorboard_log='./logs',
                     nminibatches=1)

    save_path = "./models/" + args.model + "_model"

    if args.mode == "train":
        print("Training started")
        model.learn(total_timesteps=200000, seed=69)
        model.save(save_path)
        print("Training's done, saved model to: ", save_path)
    else:
        # load pre-trained model
        print("Loading model at: ", save_path)
        model = PPO2.load(save_path)
        print("Start testing on train set")
        evaluate_train_set(model, train_env, 3000)

        if args.test_mode == 'double':
            print("Start testing on test set")
            evaluate_test_set(model, test_env, len(test_df))

        print("Testing's comeplete")






