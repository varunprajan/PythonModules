import numpy as np
import pandas as pd
import pickle
import os
import first_high_low as fhl

_UP = 1
_DOWN = -1

def win(val):
    return np.sign(val)

class Pickler(object):
    def __init__(self,dir):
        self.dir = dir

    def save(self,obj):
        fpath = self.pickle_path(obj.name)
        pickle.dump(obj,open(fpath,'wb'))
    
    def load(self,name):
        fpath = self.pickle_path(name)
        return pickle.load(open(fpath,'rb'))
    
    def pickle_path(self,name):
        fname = '{}.pkl'.format(name)
        return os.path.join(self.dir,fname)
        
class LazyFunction(object):
    def __init__(self,func):
        self.func = func
        self.val = {}
        
    def get_val(self,*args):
        try:
            return self.val[args]
        except KeyError:
            res = self.func(*args)
            self.val[args] = res
            return res
    
class TimeSeries(object):
    def __init__(self,data,name,pickler=None):
        self.data = data # of type numpy recarray
        self.name = name
        self.open = self.data['Open']
        self.close = self.data['Close']
        self.low = self.data['Low']
        self.high = self.data['High']
        self.adjclose = self.data['Adjusted Close']
        self._momentum = LazyFunction(self.gen_momentum_data)
        self._bracketreturns = LazyFunction(self.gen_bracket_returns)
        self.pickler = pickler

    # read   
    @classmethod
    def from_quandl_file(cls,filepath,name,reverse=True,**kwargs):
        dataframe = pd.read_csv(filepath,usecols=[1,2,3,4,6])
        if reverse:
            dataframe = dataframe.iloc[::-1]
        return cls(data=dataframe.to_records(),name=name,**kwargs) 
    
    # load/save
    def save(self):
        self.pickler.save(self)

    # computations over data set  
    def yield_fun(self,func,*args):
        for i, _ in enumerate(self.data):
            res = func(i,*args)
            yield res if res is not None else np.nan
        
    def momentum(self,dt):
        return self._momentum.get_val(dt)
    
    def gen_momentum_data(self,dt):
        func = self.gen_momentum_i
        return np.array([val for val in self.yield_fun(func,dt)])
    
    def gen_momentum_i(self,i,dt):
        iprev = i - 1 + dt
        if iprev >= 0:
            send = self.adjclose[i-1]
            sstart = self.adjclose[i-1+dt]
            savg = 0.5*(send + sstart)
            sdiff = send - sstart
            return sdiff/savg
    
    # bracket    
    def bracket_returns(self,rdown,rup):
        return self._bracketreturns.get_val(rdown,rup)
    
    def gen_bracket_returns(self,rdown,rup):
        func = self.bracket_trade
        return np.array([val for val in self.yield_fun(func,rdown,rup)])
    
    def bracket_trade(self,i,rdown,rup):
        currentprice = self.open[i]
        targetpriceup = currentprice*(1+rup)
        targetpricedown = currentprice*(1+rdown)
        idxup = self.hits_high(i,targetpriceup)
        idxdown = self.hits_low(i,targetpricedown)
        if idxup is not None and idxdown is not None:
            return win(idxdown - idxup)
        else:
            if idxdown is None:
                return _UP
            if idxup is None:
                return _DOWN
    
    def hits_high(self,i,target):
        idx = fhl.first_high(self.high[i:],target)
        return None if idx == -1 else idx # if idx == -1, no match
    
    def hits_low(self,i,target):
        idx = fhl.first_low(self.low[i:],target)
        return None if idx == -1 else idx # if idx == -1, no match

class BracketSimulation(object):
    def __init__(self,startmoney=100,starttime=None,endtime=None):
        self.startmoney = startmoney
        self.starttime = 0 if starttime is None else starttime
        self.endtime = -1 if endtime is None else endtime
        self.reset_simulation()
    
    def reset_simulation(self):
        self.time = self.starttime
        self.money = self.startmoney
        self.successes = 0
        self.failures = 0
    
    def simulation(self,timeseries,strategy):
        momentumdata = timeseries.momentum(strategy.dt)[self.starttime:self.endtime]
        rdown, rup = strategy.rdown, strategy.rup
        bracketdata = timeseries.bracket_returns(rdown,rup)[self.starttime:self.endtime]
        for momentum, outcome in zip(momentumdata,bracketdata):
            bet = strategy.bet(self.money,momentum=momentum)
            winnings = self.win_return(bet,outcome,rdown,rup)
            if bet != 0:
                if winnings > 0:
                    self.successes += 1
                elif winnings < 0:
                    self.failures += 1
            self.money += winnings
            self.time += 1
    
    def win_return(self,bet,outcome,rdown,rup,TOL=1e-5):
        if not np.isnan(outcome):
            success = bet*outcome
            if success > TOL: # outcome went in same direction as bet
                r = rup
            elif success < -TOL: # outcome went in opposite direction
                r = rdown
            else:
                r = 0
            return success*np.abs(r)
        else:
            return 0
    
    @property
    def roi(self):
        return (self.money - self.startmoney)/(self.time - self.starttime)
        
    @property
    def successrate(self):
        return self.successes/(self.successes + self.failures)
        
    @property
    def pupddiff(self):
        return 2*self.successrate - 1
        
class MomentumStrategy(object):
    def __init__(self,dt,momcutoff,rbounds):
        self.dt = dt
        self.momcutoffdown, self.momcutoffup = momcutoff
        self.rdown, self.rup = rbounds
            
    def bet(self,money,momentum,**kwargs):
        if not np.isnan(momentum):
            betsize = self.bet_size(money,momentum)
            if self.momcutoffdown < momentum < self.momcutoffup:
                return betsize*_UP
            else:
                return 0
        else:
            return 0

    def bet_size(self,money,momentum):
        return 0.3*money # bet constant fraction of current money