import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

import pandas as pd

import pyfolio
import empyrical

df = pd.read_csv('.\\data\\bitstamp.csv')
df = df.sort_values('Timestamp')
df.dropna(inplace=True)

slice_point =100000 # int(len(df) - 990000)

train_df = df[:slice_point]
test_df = df[slice_point:]

train_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(train_df, serial=True, commission=0)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log="./tensorboard/")


def callrender1(locals_, globals_):
    env_=locals_["self"].env
    print(env_)
    env_.recordmetrics()

net_worth_list=[]

def callrender2(locals_, globals_):
    net_worth=locals_["self"].net_worth
    net_worth_list=np.append(net_worth_list,net_worth,axis=0)


def callrender(locals_, globals_):
    """If we want to implement metrics in the environment, and call metrics in the main program, The only way is to call render in a metrics mode """
    locals_["self"].env.render(mode='metrics')


model.learn(total_timesteps=slice_point) #, callback=callrender)

train_env.render(mode='save_metrics')
train_env.close()

test=True

if test:
    test_env = DummyVecEnv(
        [lambda: BitcoinTradingEnv(test_df, serial=True)])

    obs = test_env.reset()
    for i in range(50000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        test_env.render(mode="system", title="BTC")
    
    test_env.close()
    
    test_env.render(mode='save_epoche_data')

    #test_env.render("")

