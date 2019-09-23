from src import env


class Metric:
    def __init__(self, initial_balance):
        # these properties are our metric for comparing different models
        self.win_trades = 0
        self.lose_trades = 0
        self.hold_trades = 0
        self.avg_reward = 0
        self.avg_win_value = 0
        self.avg_lose_value = 0
        self.num_step = 0
        self.most_profit_trade = 0
        self.worst_trade = 0
        self.highest_net_worth = initial_balance
        self.lowest_net_worth = initial_balance
        self.net_worth = 0
        # epoch counter, for each epoch passed (about 100k steps),
        # we will increase the epoch and add 8 more weeks to training data
        self.current_epoch = 1
        # create metrics dict for plotting purpose
        self.metrics = {"num_step": [],
                        "win_trades": [],
                        "lose_trades": [],
                        "avg_reward": [],
                        "most_profit_trade": [],
                        "worst_trade": [],
                        "net_worth": []}

    def summary(self, action, net_worth, prev_net_worth, reward, eur_held):
        profit = net_worth - prev_net_worth
        self.num_step += 1
        self.net_worth = net_worth

        if action == 0:
            self.hold_trades += 1

        if action in [env.CLOSE_AND_SELL, env.CLOSE_AND_BUY, env.CLOSE] or eur_held == 0:
            if net_worth + 10 < prev_net_worth:
                self.lose_trades += 1
                self.avg_lose_value += (1 / self.num_step * (-profit - self.avg_lose_value))
            elif net_worth > prev_net_worth:
                self.win_trades += 1
                self.avg_win_value += (1 / self.num_step * (profit - self.avg_win_value))

        self.avg_reward += reward

        self.highest_net_worth = max(net_worth, self.highest_net_worth)
        self.lowest_net_worth = min(net_worth, self.lowest_net_worth)
        self.most_profit_trade = max(profit, self.most_profit_trade)
        self.worst_trade = min(profit, self.worst_trade)

    def update_for_plotting(self):
        self.metrics["num_step"].append(self.num_step)
        self.metrics["win_trades"].append(self.win_trades)
        self.metrics["lose_trades"].append(self.lose_trades)
        self.metrics["avg_reward"].append(self.avg_reward / self.num_step)
        self.metrics["most_profit_trade"].append(self.most_profit_trade)
        self.metrics["worst_trade"].append(self.worst_trade)
        self.metrics["net_worth"].append(self.net_worth)
