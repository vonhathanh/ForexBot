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
# TODO:  
- Reorganize codes  (hafly done)
- Normalize reward and other input features  (halfly done)
- Setup prgressive training environment  
- Investigate why our win rate is 50% eventhough we did overfit the data  
- Improve model accuracy (better reward function, more features as input to our data)  
- Using news data as feature, we must calculate avg time the market is affected by 
news and then distribute it accordingly to current timeframe (halfly done)  
- Display trade history   
- Make action space more diversity  
- Write unit test  
- Use custom model  
- Implement weight initialization  
- Use convnet to test on non-stationary data (this can't be done right now because stable baseline convnet only accept image input)  
# Dones
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
