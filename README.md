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
