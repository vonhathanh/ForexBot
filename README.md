# Our step-to-step of how to build a successful trading agent  
1. Create a gym environment for our agent to learn from (env.py)  
2. Render a simple, yet elegant visualization of that environment (trading_graph.py)  
3. Train our agent to learn a profitable strategy (train.py)  
4. Evaluate our agent's performance on test data and real-time trading (test.py)  
# Requirements  
Environment: python 3.6  
Requirement: see requirements.txt  
Data source: https://forextester.com/data/datasources  
Time frame: 1 minutes (230300 <=> 23:03:00)  
# Agent design  
- Action space: buy, sell, hold.  
- Observation space: open, close, high, low, agent's networth, usd held, eur held, actions in previous 60 time step  
- Reward: networth(t) - networth(t-1)  
# TODO:  
- Improve model accuracy (better reward function, standardize data, more features as input to our data)  
- Using news data as feature, we must calculate avg time the market is affected by 
news and then distribute it accordingly to current timeframe  
- Display trade history   
- Make action space more diversity  
- Implement save and restore model functionality  
- Write unit test  
- Use custom model  
- Implement weight initialization  
- Create config parser  
- Overfit one batch of dataset to test model ability  
- Use convnet to test on non-stationary data  
# Notes:  
- Setup end-to-end training/evalution skeleton first + get dump baselines.  
- Fix random seed, start simple first, don't add too many features to our model
- Complexity only one at a time (plug signal in one by one)
- M15 time frame data contains about 200k records, we may want to switch to smaller time frame for more data  
# Dones
- Add more data (we are using data of year 2013 now)  
- Create train and test data set  
- Implement test procedure on test data set   
- Data visualization    
- Create some tools to compare performance between models  