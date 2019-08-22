# Our step-to-step of how to build a successful trading agent  
1. Create a gym environment for our agent to learn from (env.py)  
2. Render a simple, yet elegant visualization of that environment (trading_graph.py)  
3. Train our agent to learn a profitable strategy (train.py)  
4. Evaluate our agent's performance on test data and real-time trading (test.py)  
# Requirements  
- Environment: python 3.6  
- Requirement: see requirements.txt  
- Data source: https://www.histdata.com/download-free-forex-data/
- Time frame: 1 minutes (230300 <=> 23:03:00)  
# Agent design  
- Action space: buy, sell, hold.  
- Observation space: open, close, high, low, agent's networth, usd held, eur held, actions in previous 60 time step  
- Reward: networth(t) - networth(t-1)  
# Metrics to measure agent's performance
- Win/lose percent of agent, our target is 70-80% (current rate: 50-60%)
- Risk to reward ratio, target is 1:4 (the best we've achieved so far is 1:3)
- Maximum drawdown <= 0.1% of current balance (done)
- Avg win value > 0.5% is prefered
# TODO:  
- Reorganize codes  (halfly done)
- Normalize reward and other input features  (halfly done)
- Improve model accuracy (better reward function, more features as input to our data)  (doing)
- Using news data as feature, we must calculate avg time the market is affected by 
news and then distribute it accordingly to current timeframe (halfly done)  
- Display trade history   
- Make action space more diversity  
- Write unit test  
- Use custom model  
- Implement weight initialization  
- Use convnet to test on non-stationary data (this can't be done right now because stable baseline convnet only accept image input)  
# Dones
- Investigate why our win rate is 50% eventhough we did overfit the data  (done, mostly because we 
train it incorrectly)
- Setup progressive training environment  (done)
- Add close orders action, immediately release all eur agent is holding or selling  
- Split data to episodes, each episode contains 1 week data, shuffle those episodes  
- Add more data (we are using data of year 2013 now)  
- Create train and test data set  
- Implement test procedure on test data set   
- Data visualization    
- Create some tools to compare performance between models  
- Create config parser  
- Overfit one batch of dataset to test model ability  
- Implement save and restore model functionality  
# Notes:  
- Setup end-to-end training/evalution skeleton first + get dump baselines.  
- Fix random seed, start simple first, don't add too many features to our model
- Complexity only one at a time (plug signal in one by one)
- M15 time frame data contains about 200k records, we may want to switch to smaller time frame for more data  
- Right now we have reduced the spread to 0.1, in reality the spread will be 0.3 or higher,  
we need to test it against higher spread  
- Currently, we don't have any benchmark system to tell us that our agent has perform the most correct action 
(achieve highest net worth possible) in those data frames   
# Conclusions and thoughts:
- We can overfit mlp policy after 1m steps of training with data in a very short duration (300x15 minutes)  
- Eventhough we did overfit the model and it did increase our net worth overtime, but the win ratio is still 50%,  
we may need to investigate later, because our goal is win rate >= 80%  
- Win rate is not everything in Fx trading but we still need it higher 50% for more security.
- Althought our model is overfitted is still perform prety well on test set: x6 networth in about 6000 steps,  
It also go bankrupt in 1-2000 steps later on which is the result that we expected.  
- Our objective is increase the avg win value per trade and win ratio as well as reduce the avg lose per trade  
=> First conclusion: more data is not really good for model if we don't understand and arrange the data correctly, 
on the other hand, more training is really important in RL  

- Agent sometimes buy or sell eur in agressive manner, it performs the same order in many continous steps. Which 
is why it loses or wins an immense amount of money, but it's not really what we want.  
- We would like the agent to control the risk-to-reward ratio that maximum drawdown is as low as possible  
- However, it means that we won't win big amount because we don't allow the agent to invest too much in one currency  
- This is some kind of trade-off and we rather play safe than put our capital at risk.  
=> We need a reward function that punish agent for holding too much eur.  
i.e. function that take into account the risk of producing high return  
=> Our model learn to hold a numerous amount of eur in order to achieve great return.   
However, this strategy is a double-edge sword, we could go bankrupt because of this  
So, great rewards doesn't have to come from great returns, we could reward the agent when it wins small
consecutive steps and not losing continously  

- We have two kinds of agent at the moment: one has 50% win rate, avg win/lose value is 100/30 and the other
 win rate is 60%, avg win/lose is 30/10  
=> This means agent has to sacrifice win rate to increase win/lose value.  

