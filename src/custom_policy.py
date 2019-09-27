from stable_baselines.common.policies import MlpLstmPolicy


class CustomLSTMPolicy(MlpLstmPolicy):
    def __init__(
            self,
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm=64,
            reuse=False,
            **_kwargs):
        super(
            MlpLstmPolicy,
            self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            layer_norm=False,
            feature_extraction="mlp",
            net_arch=['lstm', dict(pi=[512, 256, 64],
                                   vf=[512, 256, 64])],
            **_kwargs)
