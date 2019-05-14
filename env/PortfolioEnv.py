import gym
import pandas as pd
import numpy as np

import datetime as dt

from gym import spaces
from sklearn import preprocessing

import pyfolio

#from pyfolio import plotting
#from pyfolio import  perf_attrib
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


class PortfolioEnv(gym.Env):
    """A trading environment with porfolio function for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none','metrics', 'show_metrics','save_all_metrics']} # I added a metrics mode
    scaler = preprocessing.MinMaxScaler() # Todo: replace this for it looks ahead

    def __init__(self, 
                    df_price, # The origial PRICE dataframe
                    lookback_window_size=40, # The lookback window size. The history to use to make decision
                    initial_balance=10000, # The initial balance
                    commission=0.00075, # The commission rate
                    epoche_size=40 # walk ahead vs lookback
                 ):
        super(PortfolioEnv, self).__init__()

        self.state_values=df_price.pct_change().to_numpy()[2:,:]
        self.df_price = df_price.iloc[2:,:]   #Todo: the dataframe should be preprocessed in advance

        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.epoche_size = epoche_size
        self._epoche_steps = 0

        # Actions of the format Buy 1/10, Sell 3/10, Hold (amount ignored), etc.
        self.action_space = spaces.MultiDiscrete([3, 10])

        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, lookback_window_size + 1), dtype=np.float16)

        self.metrics_history=[] # Todo: we need to make array of dictionary 

        self._reset_portfolio()

    def _next_observation(self):
        """Get the next observation on the market"""
        end = self.current_step + self.lookback_window_size + 1

        obs =  np.transpose(self.state_values[self.current_step:end])
       
        return obs

    def _reset_session(self):
        """Reset the observation on the market"""
        self.current_step = 0

        self.steps_left = self.epoche_size
        self.frame_start = np.random.randint(
                self.lookback_window_size+1, len(self.df_price) - self.steps_left)

        self.active_df = self.df_price[self.frame_start - self.lookback_window_size:
                                 self.frame_start + self.steps_left]

        self.active_df = self.scaler.fit_transform(self.active_df)
        self.active_df = pd.DataFrame(
            self.active_df, columns=self.df_price.columns)
    
    def _reset_portfolio(self):
        """Reset the portfolio for the epoche"""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        self.previous_net_worth = self.initial_balance

        self.btc_held = 0

        self.account_history = np.repeat([
            [self.net_worth],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)
        self.trades = []
        self.done = False

        self.returns = []

    def reset(self):
        """Reset the environment.   
        1. reset the observation on the market
        2. reset the portfolio """
       
        self._reset_session()
        self._reset_portfolio()
     
        return self._next_observation()

    def _get_current_price(self):
        return self.df_price['Close'].values[self.frame_start + self.current_step]

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

        obs = self._next_observation()
        reward = self.net_worth - self.previous_net_worth # I substracted previous value

        self.steps_left -= 1
        self.current_step += 1

        self.done = (self.net_worth <= 0) | (self.steps_left==0)
        if self.done:
            # Todo 1. To save the epoche metrics to the environmnt -- System level
            # Todo 2. To call user function to save the epoche data 
            #self.save_epoche_data()
            self.save_epoche_metrics()

        #if self.steps_left == 0:
        #    self.balance += self.btc_held * current_price
        #    self.btc_held = 0

        #    self._reset_session()

        #Todo: To make reward use metrics
            
        return obs, reward, self.done, {}

    def render(self, mode='human', **kwargs):
        if mode == 'system':
            print('Price: ' + str(self._get_current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step ]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step ]))
            print('Net worth: ' + str(self.net_worth))

        elif mode == 'metrics':
            #Todo to call pyfolio to generate self.save_metrics()
            if self.done: # Only when the epoche is done
                #print('Net worth: ' + str(self.net_worth))
                self.show_epoche_metrics()

        elif mode == 'show_metrics':
            self.show_epoche_metrics()
        elif mode == 'save_all_metrics':
            self.save_all_metrics()
            

    def clear_metrics(self):# To use pyfolio
        self.returns=[]

    def save_epoche_metrics(self):
        print('Net worth: ' + str(self.net_worth)) # for debug
        metrics=self.get_metrics()

        self.metrics_history.append(metrics)

    def save_all_metrics(self):
        df=pd.DataFrame.from_dict(self.metrics_history, orient='columns')
        df.to_csv(".\\metrics\\all_metrics.csv")

    def save_epoche_trades(self):
        pd.DataFrame(self.trades).to_csv(".\\metrics\\trades_"+str(dt.datetime.now())+".csv")
        #print("save_epoche_data"+str(dt.datetime.now()))
 
    def show_epoche_metrics(self):
        dti=pd.date_range('2018-01-01', periods=self.returns.shape[0], freq='D')
        ps=pd.Series(self.returns,dtype="float64",index=dti)
        #plotting.show_perf_stats(ps,ps)
        create_returns_tear_sheet(ps,benchmark_rets=ps)
        #print(self.get_metrics())
 
    def close(self):
        #self.save_all_metrics() #Todo To save all metrics of epoches
        print("close")

    def get_metrics(self):
        SIMPLE_STAT_FUNCS = [
    #ep.annual_return,
    ep.cum_returns_final,
    #ep.annual_volatility,
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
        if self.net_worth==10000.0:
            print('Why?')

        stats = {} # pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            #stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(self.returns)
            stats[stat_func.__name__] = stat_func(self.returns)

        return stats
