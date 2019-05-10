# This is an OPENAI gym reinforcement learning (RL) environment
This is an OPENAI gym reinforcement learning environment for investing. There are portfolio performance and risk metrics functions built in. These functions can be used to evaluate your algorithm and also in reward engineering.

So that this gym totally compatible to OPENAI gym (https://github.com/openai/gym)

# Not just another gym for investing

I tried a lot of gym. They are only a general gym equiped with investment data (as STATE of RL environment).

For investing, the basic function is the performance and risk metrics. Without that it's hard to know wheather the algo is good or not. 

Also, in most situations, the profit is not the target objectives. Usually, the risk adjusted return or some ratios are the real one instead.

So we need performance and risk metrics built in, not only as STATE, but also as REWARD of RL environment.

# In future

This is the first step to make an investing specific gym.
Besides Performance and Risk Metrics, We need more investing specific functions built-in.

1. Portfolio Management (PnL, Turnover, Weights)
2. Transaction Management (Commision, slippage)
3. External Price Sources support

# Credits
1. OPENAI gym (https://github.com/openai/gym)
2. Stock-Trading-Environment (https://github.com/notadamking/Stock-Trading-Environment)
3. Pyfolio (https://github.com/quantopian/pyfolio)
