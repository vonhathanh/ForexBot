# Our step-to-step of how to build a successful trading agent  
1. Create a gym environment for our agent to learn from (env.py)  
2. Render a simple, yet elegant visualization of that environment (render.py)  
3. Train our agent to learn a profitable strategy (train.py)  
4. Evaluate our agent's performance on test data and real-time trading (test.py)  

Environment: python 3.6  
Requirement: see requirements.txt  
Data source: https://forextester.com/data/datasources  
Time frame: 15 minutes (230300 <=> 23:03:00)
  
# Agent design  
- action space: buy, sell, hold.  
- observation space: open, close, high, low, agent's networth, usd held, eur held  in previous 60 time step  
- Reward: networth(t) - networth(t-1)
