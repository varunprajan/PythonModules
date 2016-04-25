import numpy as np
# import first_high_low as fhl
import finance_base as fbase

_UP = 1
_DOWN = -1

class MomentumTimeSeries(fbase.TimeSeries):
    def __init__(self,data,name,adjclosekey,lowkey=None,highkey=None,pickler=None,startdate=None,enddate=None):
        super().__init__(data,name,adjclosekey,startdate=startdate,enddate=enddate)
        self.low = self.data[lowkey] if lowkey is not None else None
        self.high = self.data[highkey] if highkey is not None else None
        self._momentum = fbase.LazyFunction(self.gen_momentum_data)
        self._bracketreturns = fbase.LazyFunction(self.gen_bracket_returns)
        self.pickler = pickler
    
    # momentum
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
        return np.array([list(val) for val in self.yield_fun(func,rdown,rup)])
    
    def bracket_trade(self,i,rdown,rup):
        currentprice = self.adjclose[i-1]
        targetpriceup = currentprice*(1+rup)
        targetpricedown = currentprice*(1+rdown)
        idxup = self.hits_high(i,targetpriceup)
        idxdown = self.hits_low(i,targetpricedown)
        if idxup is not None and idxdown is not None:
            if idxdown == idxup:
                return 0, 0
            elif idxdown > idxup:
                return _UP, idxup
            else:
                return _DOWN, idxdown
        else:
            if idxdown is not None:
                return _DOWN, idxdown
            if idxup is not None:
                return _UP, idxup
            return np.nan, 0
    
    def hits_high(self,i,target):
        # idx = fhl.first_high(self.high[i:],target)
        return None if idx == -1 else idx # if idx == -1, no match
    
    def hits_low(self,i,target):
        # idx = fhl.first_low(self.low[i:],target)
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
        self.simulation_sub(strategy,momentumdata,bracketdata,rdown,rup)
        
    def simulation_sub(self,strategy,momentumdata,bracketdata,rdown,rup):
        winningsvec = np.zeros(momentumdata.shape)
        n = winningsvec.shape[0]
        for i, (momentum, bracket) in enumerate(zip(momentumdata,bracketdata)):
            bet = strategy.bet(self.money,momentum=momentum)
            if bet != 0:
                outcome, idx = bracket # outcome occurs at an index idx w.r.t. i
                winnings = self.win_return(bet,outcome,rdown,rup)
                if winnings > 0:
                    self.successes += 1
                elif winnings < 0:
                    self.failures += 1
                winningtime = i + idx
                if winningtime < n: # do not count winning events that occur after end time
                    winningsvec[winningtime] += winnings
            self.money += winningsvec[i]
        self.time += n
    
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
    
    def annual_roi(self,timeseries,DAYFAC=365):
        timediff = timeseries.date[self.time] - timeseries.date[self.starttime]
        nyears = timediff.days/DAYFAC
        return (self.money/self.startmoney)**(1/nyears) - 1
        
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
        return 1.0*money # bet constant fraction of current money