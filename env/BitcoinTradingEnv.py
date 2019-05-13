import gym
import pandas as pd
import numpy as np

import datetime as dt

from gym import spaces
from sklearn import preprocessing

import pyfolio

from pyfolio import plotting
from pyfolio import  perf_attrib
import math
import empyrical as ep

from pyfolio.tears import (create_full_tear_sheet,
                           create_simple_tear_sheet,
                           create_returns_tear_sheet,
                           create_position_tear_sheet,
                           create_txn_tear_sheet,
                           create_round_trip_tear_sheet,
                           create_interesting_times_tear_sheet,
                           create_bayesian_tear_sheet)

MAX_TRADING_SESSION = 1000


class BitcoinTradingEnv(gym.Env):
    """A Bitcoin trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none','metrics', 'save_metrics','save_epoche_data']} # I added a metrics mode
    scaler = preprocessing.MinMaxScaler()

    def __init__(self, df, lookback_window_size=40, initial_balance=10000, commission=0.00075, serial=False,
                 epoche_size=40, # walk ahead vs lookback
                 ):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial
        self.epoche_size = epoche_size #
        self._epoche_steps = 0

        self.previous_net_worth = self.initial_balance

        # Actions of the format Buy 1/10, Sell 3/10, Hold (amount ignored), etc.
        self.action_space = spaces.MultiDiscrete([3, 10])

        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10, lookback_window_size + 1), dtype=np.float16)

        self.net_worth_list=[]
        self.returns=[]

        self.metrics_history=[]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1

        obs = np.array([
            self.active_df['Open'].values[self.current_step:end],
            self.active_df['High'].values[self.current_step:end],
            self.active_df['Low'].values[self.current_step:end],
            self.active_df['Close'].values[self.current_step:end],
            self.active_df['Volume_(BTC)'].values[self.current_step:end],
        ])

        scaled_history = self.scaler.fit_transform(self.account_history)

        obs = np.append(
            obs, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)

        return obs

    def _reset_session(self):
        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.lookback_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.lookback_window_size:
                                 self.frame_start + self.steps_left]
        self.active_df = self.scaler.fit_transform(self.active_df)
        self.active_df = pd.DataFrame(
            self.active_df, columns=self.df.columns)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        self.previous_net_worth = self.initial_balance

        self.btc_held = 0

        self._reset_session()

        self.account_history = np.repeat([
            [self.net_worth],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)
        self.trades = []
        self.done = False

        self.net_worth_list = []
        self.returns = []
        return self._next_observation()

    def _get_current_price(self):
        return self.df['Close'].values[self.frame_start + self.current_step]

    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * (current_price + 0.01) * (1 + self.commission) # the bid-ask spread, 0.01, should be here

            self.btc_held += btc_bought
            self.balance -= cost

        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * (current_price - 0.01) * (1 - self.commission) # the bid-ask spread, 0.01, should be here

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'action_0': action[0],'action_1': action[1],
                                'type': "sell" if btc_sold > 0 else "buy"})

        self.previous_net_worth = self.net_worth
        self.net_worth = self.balance + self.btc_held * current_price

        self.returns=np.append(self.returns, math.log(self.net_worth/self.previous_net_worth))

        #What, meaningless noise was added!
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def step(self, action):
        current_price = self._get_current_price() # why + 0.01? here

        self._take_action(action, current_price)

        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0

            self._reset_session()

        obs = self._next_observation()
        reward = self.net_worth - self.previous_net_worth # I substracted previous value
        self.done = (self.net_worth <= 0) | (self.current_step>=self.epoche_size)

        if self.done:
            # Todo 1. To save the epoche metrics to the environmnt -- System level
            # Todo 2. To call user function to save the epoche data 
            #self.save_epoche_data()
            self.save_metrics()
            self.returns=[]

        return obs, reward, self.done, {}

    def render(self, mode='human', **kwargs):
        if mode == 'system':
            print('Price: ' + str(self._get_current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step + self.frame_start]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step + self.frame_start]))
            print('Net worth: ' + str(self.net_worth))

        elif mode == 'metrics':
            #Todo to call pyfolio to generate self.save_metrics()
            if self.done: # Only when the epoche is done
                print('Net worth: ' + str(self.net_worth))
        elif mode == 'show_metrics':
            self.show_epoche_metrics()
            

    def clear_metrics(self):# To use pyfolio
        self.net_worth_list=[]
        self.returns=[]

    def save_metrics(self):
        print('Net worth: ' + str(self.net_worth))
        #self.net_worth_list=np.append(self.net_worth_list,[self.net_worth],axis=0)
        #pd.DataFrame({"net_worth_list":self.net_worth_list}).to_csv("./metrics/net_worth_list.csv")
        metrics=self.get_metrics()

        self.metrics_history= np.append(self.metrics_history, metrics)

    def clear_epoche_data(self):
        self.trades = []

    def save_epoche_data(self):
        #pd.DataFrame(self.trades).to_csv(".\\metrics\\trades_"+str(dt.datetime.now())+".csv")
        print("save_epoche_data"+str(dt.datetime.now()))
 
    def show_epoche_metrics(self):
        dti=pd.date_range('2018-01-01', periods=self.returns.shape[0], freq='D')
        ps=pd.Series(self.returns,dtype="float64",index=dti)
        #plotting.show_perf_stats(ps,ps)
        #create_returns_tear_sheet(ps,benchmark_rets=ps)
        print(self.get_metrics())
 
    def close(self):
        self.save_metrics() #Todo To save all metrics of epoches
        print("close")

    def get_metrics(self):
        SIMPLE_STAT_FUNCS = [
    ep.annual_return,
    ep.cum_returns_final,
    ep.annual_volatility,
    ep.sharpe_ratio,
    ep.calmar_ratio,
    ep.stability_of_timeseries,
    ep.max_drawdown,
    ep.omega_ratio,
    ep.sortino_ratio #,
    #stats.skew,
    #stats.kurtosis,
    #ep.tail_ratio,
    #value_at_risk
]
        STAT_FUNC_NAMES = {
    'annual_return': 'Annual return',
    'cum_returns_final': 'Cumulative returns',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'calmar_ratio': 'Calmar ratio',
    'stability_of_timeseries': 'Stability',
    'max_drawdown': 'Max drawdown',
    'omega_ratio': 'Omega ratio',
    'sortino_ratio': 'Sortino ratio',
    'skew': 'Skew',
    'kurtosis': 'Kurtosis',
    'tail_ratio': 'Tail ratio',
    'common_sense_ratio': 'Common sense ratio',
    'value_at_risk': 'Daily value at risk',
    'alpha': 'Alpha',
    'beta': 'Beta',
}

        stats = pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(self.returns)

        return stats


