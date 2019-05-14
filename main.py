import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.PortfolioEnv import PortfolioEnv

import pandas as pd

#import pyfolio
import empyrical

df = pd.read_csv('.\\data\\bitstamp.csv',parse_dates=['Timestamp'],index_col=['Timestamp'])
df.dropna(inplace=True)
df = df.reset_index()
df=df[["Open","High","Low","Close","Timestamp"]]
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)

df["Open"]=df["Open"].astype("float64")
df["High"]=df["High"].astype("float64")
df["Low"]=df["Low"].astype("float64")
df["Close"]=df["Close"].astype("float64")

slice_point = int(len(df) - 10000) # for debug

train_df = df[:slice_point]
test_df = df[slice_point:]

train_env = DummyVecEnv(
    [lambda: PortfolioEnv(train_df, commission=0)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log="./tensorboard/")

def callrender(locals_, globals_):
    """If we want to implement metrics in the environment, and call metrics in the main program, The only way is to call render in a metrics mode """
    locals_["self"].env.render(mode='metrics')


model.learn(total_timesteps=slice_point, callback=callrender)

train_env.render(mode='save_all_metrics')
# train_env.save_all_metrics() # This doesn't work. We have to use the above

train_env.close()

test=True

if test:
    test_env = DummyVecEnv(
        [lambda: PortfolioEnv(test_df)])

    obs = test_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        test_env.render(mode="metrics", title="BTC")
        if done:
            break
    
    test_env.close()
