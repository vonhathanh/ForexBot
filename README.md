# Our step-to-step of how to build a successful trading agent  
1. Create a gym environment for our agent to learn from (env.py)  
2. Render a simple, yet elegant visualization of that environment (render.py)  
3. Train our agent to learn a profitable strategy (train.py)  
4. Evaluate our agent's performance on test data and real-time trading (trading_graph.py)  
  
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
- Add more data (we are using data of year 2013 now) (done)  
- Create train and test data set (done)  
- Implement test procedure on test data set  
- Using news data as feature  
- Display trade history  
- Saving training process to tensorboard for later evaluation  
- Make action space more diversity  
- Implement save model functionality  
- Improve accuracy  
- Write unit test  
  
# Notes:  
- M15 time frame data contains about 200k records, we may want to switch to smaller time frame for more data  
