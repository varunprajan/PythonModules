from decimal import Decimal
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
import datetime
import finance_base as fbase
import copy

TOL = 1e-10
TOLCASH = Decimal(0.00001)
ZERO = Decimal(0)

class Simulation(object):
    def __init__(self,startdate,enddate,timeseries,allportfolios,marginrate,capitalgainsrate=0.2,commission=0.005):
        self.nbuff = 100
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate)
        self.timeseries = timeseries
        self.allportfolios = allportfolios
        self.marginrate = Decimal(marginrate)
        self.capitalgainsrate = Decimal(capitalgainsrate)
        self.allportfolios.commission = Decimal(commission)
        self.init_idx_dates()
        self.init_performance()
  
    def init_idx_dates(self):
        self.idx = {}
        for name, series in self.timeseries.items():
            startidx = series.idx_closest_date(self.startdate)
            endidx = series.idx_closest_date(self.enddate)
            self.idx[name] = (startidx,endidx)
        idxdiff = [idxpair[1] - idxpair[0] for idxpair in self.idx.values()]
        if len(set(idxdiff)) > 1:
            raise ValueError('Different date series appear to have disregistry')
        self.idxdiff = idxdiff[0]
    
    def init_performance(self):
        self.performance = [np.empty((self.idxdiff+self.nbuff,len(plist)+1)) for plist in self.allportfolios.assetlist]            
    
    def init_cash(self,cash):
        for portfolio in self.allportfolios.portfoliolist:
            portfolioname = portfolio.name
            self.allportfolios.deposit_or_withdraw_cash(portfolioname,cash,date=0)
    
    def run_simulation(self,startingcash):
        self.prevmonth = 0
        self.get_and_set_prices(counter=0)
        self.init_cash(startingcash)
        for counter in range(self.idxdiff):
            self.get_and_set_prices(counter)
            self.write_to_performance(counter)
            self.allportfolios.update_all(date=counter)
            if self.is_at_start_of_month(counter):
                self.allportfolios.apply_margin(self.marginrate)
        self.allportfolios.liquidate(self.capitalgainsrate,date=self.idxdiff)
        for i in range(self.nbuff):
            self.write_to_performance(counter=self.idxdiff + i)
        
    def write_to_performance(self,counter):
        for performance, portfolio, plist in zip(self.performance,self.allportfolios.portfoliolist,self.allportfolios.assetlist):
            for i, assetname in enumerate(plist):
                performance[counter,i] = portfolio._asset_value(assetname)
            performance[counter,-1] = portfolio.cash
        
    def is_at_start_of_month(self,counter):
        randomassetname = self.allportfolios.assetlist[0][0]
        idxcurr = self.idx[randomassetname][0] + counter
        date = self.timeseries[randomassetname].date[idxcurr]
        prevmonth = self.prevmonth
        self.prevmonth = date.month
        return (date.month != prevmonth)
            
    def get_and_set_prices(self,counter):
        prices = {}
        for assetname in self.allportfolios.assetnames:
            idxcurr = self.idx[assetname][0] + counter
            prices[assetname] = Decimal(self.timeseries[assetname].adjclose[idxcurr])
        self.allportfolios.set_prices(prices)
    
    # plotting
    def individual_performance_plot(self,portfolionum):
        xplot = self.time_plot()
        return [np.column_stack((xplot,yplot)) for yplot in self.performance[portfolionum].T]
        
    def total_performance_plot(self,portfolionum):
        xplot = self.time_plot()
        yplot = np.sum(self.performance[portfolionum],axis=1)
        return [np.column_stack((xplot,yplot))]
        
    def time_plot(self):
        return np.arange(self.performance[0].shape[0])
        
    def yearly_return(self,portfolionum,DAYFAC=365):
        final = float(self.allportfolios.portfoliolist[portfolionum]._equity)
        initial = self.asset_curr(portfolionum,0)
        timediff = self.enddate - self.startdate
        nyears = timediff.days/DAYFAC
        return (final/initial)**(1/nyears) - 1
        
    def asset_curr(self,portfolionum,counter):
        return np.sum(self.performance[portfolionum][counter,:])

class AllPortfolios(object):
    def __init__(self,commission=0.0):
        self.portfoliolist = []
        self.assetnames = set()
        self.assetlist = []
        self.commission = Decimal(commission)
    
    def set_prices(self,prices):
        for portfolio in self.portfoliolist:
            portfolio.currentprices = prices
    
    def add_portfolio_from_props(self,name,fdesired,fdriftmax,Lmin,Lmax,Ltarget):
        newportfolio = BalancedPortfolio(name,fdesired,fdriftmax,Lmin,Lmax,Ltarget)
        self.add_portfolio(newportfolio)
        
    def add_portfolio(self,newportfolio):
        self.portfoliolist.append(newportfolio)
        self.add_assetnames_from_portfolio(newportfolio)
        
    def add_assetnames_from_portfolio(self,portfolio):
        assetnames = set(portfolio.fdesired.keys())
        self.assetnames = self.assetnames.union(assetnames)
        self.assetlist.append(list(assetnames))
    
    def deposit_or_withdraw_cash(self,portfolioname,value,date=None):
        self._modify_portfolio_by_name('deposit_or_withdraw_cash',portfolioname,date,value=value)
        
    def assign_new_allocation(self,portfolioname,fdesired,date=None):
        self._modify_portfolio_by_name('assign_new_allocation',portfolioname,date,fdesired=fdesired)
        
    def set_leverage_limits(self,portfolioname,Lmin,Lmax,Ltarget,date=None):
        self._modify_portfolio_by_name('set_leverage_limits',portfolioname,date,Lmin=Lmin,Lmax=Lmax,Ltarget=Ltarget)
    
    def update_portfolio(self,portfolio,date=None):
        self._modify_portfolio('update',portfolio,date)
    
    def _modify_portfolio_by_name(self,methodname,portfolioname,date,**kwargs):
        portfolio = self.get_portfolio_by_name(portfolioname)
        self._modify_portfolio(methodname,portfolio,date,**kwargs)
    
    def _modify_portfolio(self,methodname,portfolio,date,**kwargs):
        if date is None:
            date = datetime.datetime.now()
        fun = getattr(portfolio,methodname) # get portfolio method
        fun(**kwargs) # do stuff (deposit, set allocation, etc.)
        self.execute_orders_from_portfolio(portfolio,date) # buy with new money, or sell to get withdrawal money
        
    def get_portfolio_by_name(self,name):
        for portfolio in self.portfoliolist:
            if portfolio.name == name:
                return portfolio
        raise ValueError('Portfolio name not found')
        
    def update_all(self,date=None):
        for portfolio in self.portfoliolist:
            self.update_portfolio(portfolio,date)
    
    def execute_orders_from_portfolio(self,portfolio,date):
        actualprices = {}
        for assetname, amount in portfolio.currentorders.items():
            # insert order here, get price
            # print('Ordering {} shares of {}'.format(amount,assetname))
            actualprices[assetname] = portfolio.currentprices[assetname] # TODO: FIX!
            portfolio.cash -= self.commission*abs(amount)
        portfolio.log_orders_and_correct_cash(actualprices,date)
        portfolio.reset_current_orders()
    
    def apply_margin(self,marginrate):
        for portfolio in self.portfoliolist:
            if portfolio.cash < ZERO:
                portfolio.cash *= 1 + marginrate
                
    def liquidate(self,capitalgainsrate,date=None):
        for portfolio in self.portfoliolist:
            self._modify_portfolio('liquidate',portfolio,date)
            portfolio.enforce_capital_gains(capitalgainsrate)   
    
    # for resolving discrepancy with actual account
    def total_cash(self):
        return sum(portfolio.cash for portfolio in self.portfoliolist)
    
    def correct_for_total_cash_diff(self,actualcash):
        assumedtotalcash = self.total_cash()
        totalcashdiff = actualcash - assumedtotalcash
        print('Total cash difference: {}'.format(totalcashdiff))
        for portfolio in portfoliolist:
            weight = portfolio.cash/assumedcash
            portfoliocashdiff = totalcashdiff*weight
            portfolio.modify_cash(portfoliocashdiff)          
        
class BalancedPortfolio(object):
    def __init__(self,name,fdesired,fdriftmax,Lmin,Lmax,Ltarget):
        self.name = name
        self.assets = {}
        self.cash = ZERO
        self.fdriftmax = fdriftmax
        self.currentprices = {}
        self.reset_current_orders()
        self.transactionhistory = []
        self.set_leverage_limits(Lmin,Lmax,Ltarget,reset=False)
        self.assign_new_allocation(fdesired,reset=False)
        
    @classmethod
    def test_portfolio(cls,fstocks=None,fbonds=None,fgold=None,name='try',fdriftmax=0.05,Lmin=1.45,Lmax=1.55,Ltarget=1.5):
        fdesired = {}
        if fstocks is not None and fstocks > TOL:
            fdesired['Stocks'] = fstocks
        if fbonds is not None and fbonds > TOL:
            fdesired['Bonds'] = fbonds
        if fgold is not None and fgold > TOL:
            fdesired['Gold'] = fgold
        return cls(name,fdesired,fdriftmax,Lmin,Lmax,Ltarget)
    
    # public methods    
    def update(self):
        self._order_allocated_assets_using_cash() # if there is available cash (e.g., no leverage, from dividends), buy assets       
        self._check_and_correct_leverage()
        self._check_and_rebalance_assets()
    
    def view(self,verbose=True):
        print(self._stringify(verbose))

    def assign_new_allocation(self,fdesired,reset=True):
        self._check_and_set_fdesired(fdesired)
        self._add_and_delete_assets()
        if reset:
            self.update() # need to buy assets with available cash, correct leverage, rebalance, etc.             
            
    def set_leverage_limits(self,Lmin,Lmax,Ltarget,reset=True):
        self._check_L_limits(Lmin,Lmax,Ltarget)
        self.Lmin = Decimal(Lmin)
        self.Lmax = Decimal(Lmax)
        self.Ltarget = Decimal(Ltarget)
        if reset:
            self.check_and_correct_leverage()

    def deposit_or_withdraw_cash(self,value):
        self._check_equity(value)
        self.modify_cash(value)
        self._order_allocated_assets(value)
        self._check_and_correct_leverage()
        
    def liquidate(self):
        for assetname in self.fdesired.keys():
            self._liquidate_asset(assetname)
    
    def enforce_capital_gains(self,capitalgainsrate):
        gains = self._naive_capital_gains()
        self.modify_cash(-capitalgainsrate*Decimal(gains))
        
    def log_orders_and_correct_cash(self,actualprices,date):
        self._correct_cash(actualprices)
        self._log_transactions(actualprices,date)
        
    def reset_current_orders(self):
        self.currentorders = defaultdict(int)
        
    def modify_cash(self,delta):
        self.cash += Decimal(delta)
    
    # view portfolio
    def __str__(self):
        return self._stringify(verbose=False)
    
    def _stringify(self,verbose):
        def yield_str():
            yield 'Portfolio {}'.format(self.name)
            yield 'Equity: ${:.2f}'.format(self._equity)
            yield 'Leverage ratio: {:.3f}'.format(self._leverage_ratio)
            if verbose:
                yield 'Asset allocation:'
                for assetname, amount in self.assets.items():
                    yield '\t{}: ${:.2f}'.format(assetname,self._asset_value(assetname))
                yield 'Cash: ${:.2f}'.format(self.cash)
        return '\n'.join(string for string in yield_str())

    # basic properties
    @property
    def _leverage_ratio(self):
        cashfudge = self.cash if self.cash > 0 else 0
        return (self._portfolio_value + cashfudge)/self._equity
    
    @property
    def _portfolio_value(self):
        return sum(self._asset_value(assetname) for assetname in self.assets.keys())
          
    def _asset_value(self,assetname):
        return Decimal(self.assets[assetname])*self.currentprices[assetname]

    @property
    def _equity(self):
        return self._portfolio_value + self.cash
    
    # deposits, withdrawals, borrowing           
    def _check_equity(self,delta):
        if self._equity + Decimal(delta) < ZERO:
            raise ValueError('Cannot withdraw such a large amount: insufficient equity')    
    
    # leverage    
    def _check_L_limits(self,Lmin,Lmax,Ltarget):
        if Lmin < 1:
            raise ValueError('All leverage limits must be at least 1')
        if not (Lmin <= Ltarget <= Lmax):
            raise ValueError('Lmin must be less than or equal to Ltarget; Ltarget must be less than or equal to Lmax')
    
    def _check_and_correct_leverage(self):
        Lcurr = self._leverage_ratio
        if self._bad_leverage(Lcurr):
            Lnew = self._target_leverage(Lcurr)
            deltaborrow = self._amount_to_borrow(Lcurr,Lnew)
            self._order_allocated_assets(deltaborrow)
    
    def _bad_leverage(self,Lcurr):
        return not (self.Lmin <= Lcurr <= self.Lmax)
        
    def _amount_to_borrow(self,Lcurr,Lnew):
        return (Lnew-Lcurr)*self._equity
        
    def _target_leverage(self,Lcurr):
        return self.Ltarget       

    # drift/rebalancing
    def _check_and_set_fdesired(self,fdesired):
        fdesired = {assetname: Decimal(f) for assetname, f in fdesired.items()}
        ftotal = sum(fdesired.values())
        if abs(ftotal - 1) > TOL:
            raise ValueError('Desired fs of portfolio do not add up to 1')
        self.fdesired = fdesired
        
    def _add_and_delete_assets(self):
        oldassets = set(self.assets.keys())
        newassets = set(self.fdesired.keys())
        addedassets = newassets.difference(oldassets)
        deletedassets = oldassets.difference(newassets)
        for assetname in deletedassets:
            self._liquidate_asset(assetname)
            del self.assets[assetname]
        for assetname in addedassets:
            self.assets[assetname] = 0
    
    def _check_and_rebalance_assets(self):
        if self._f_drift > self.fdriftmax:
            self._rebalance_assets()        
        
    def _rebalance_assets(self):
        portfoliovalue = self._portfolio_value
        trades = ((assetname, portfoliovalue*fdiff) for assetname, fdiff in self._yield_f_diff()) # order portfoliovalue*(fdesired-factual) worth of assets
        tradessorted = sorted(trades, key = lambda x: x[1]) # sell assets before buying others in case change is large
        self._order_assets(tradessorted)
            
    @property
    def _f_drift(self):
        return sum(abs(fdiff) for _, fdiff in self._yield_f_diff())
        
    def _yield_f_diff(self):
        portfoliovalue = self._portfolio_value
        for assetname, fdesired in self.fdesired.items():
            factual = self._f_actual(assetname,portfoliovalue)
            yield assetname, fdesired - factual
    
    def _f_actual(self,assetname,portfoliovalue):
        return self._asset_value(assetname)/portfoliovalue
    
    # sell, buy assets    
    def _order_allocated_assets_using_cash(self):
        if self.cash > ZERO:
            self._order_allocated_assets(self.cash)
    
    def _order_allocated_assets(self,total):
        orders = self._allocated_orders(total)
        self._order_assets(orders)
    
    def _allocated_orders(self,total):
        return [(assetname, total*fdesired) for (assetname, fdesired) in self.fdesired.items()]
    
    def _liquidate_asset(self,assetname):
        currentholding = self._asset_value(assetname)
        self._order_asset(assetname,-currentholding)
    
    def _order_assets(self,orders):
        for assetname, value in orders:
            self._order_asset(assetname,value)           
    
    def _order_asset(self,assetname,value):
        price = self.currentprices[assetname]
        amount = int(round(value/price)) # can only buy whole shares
        self.modify_cash(-amount*price)
        self._modify_asset(assetname,amount)
        self._add_order(assetname,amount)
        # print('Ordering asset {} in amount ${}'.format(assetname,amount))
    
    def _add_order(self,assetname,amount):
        self.currentorders[assetname] += amount
        
    def _modify_asset(self,assetname,deltaamount):
        """Change asset amount, enforcing the condition that shorting is not allowed"""
        if self.assets[assetname] + deltaamount < 0:
            raise ValueError('Current holding in {} is {:.2f}, while requested change is {:.2f}'.format(assetname,currentamount,deltaamount))
        self.assets[assetname] += deltaamount
    
    # correction using actual prices
    def _correct_cash(self,actualprices):
        cashdiff = ZERO
        for assetname, amount in self.currentorders.items():
            assumedprice = self.currentprices[assetname]
            actualprice = actualprices[assetname]
            cashdiff += amount*(assumedprice - actualprice)
        self.modify_cash(cashdiff)
    
    # transaction logging/capital gains
    def _log_transactions(self,actualprices,date):
        for assetname, amount in self.currentorders.items():
            actualprice = actualprices[assetname]
            newtransaction = fbase.Transaction(date,assetname,amount,actualprice)
            self.transactionhistory.append(newtransaction)

    def _naive_capital_gains(self):
        gains = 0
        temp = copy.deepcopy(self.transactionhistory)
        for assetname in self.assets.keys():
            buys = [transaction for transaction in temp if (transaction.shares > 0 and transaction.assetname == assetname)]
            buys = sorted(buys, key = lambda x: x.time)
            sells = [transaction for transaction in temp if (transaction.shares < 0 and transaction.assetname == assetname)]
            sells = sorted(sells, key = lambda x: x.time)
            counterbuy, countersell = 0, 0
            while (counterbuy < len(buys)) and (countersell < len(sells)):
                nsharesbuy = buys[counterbuy].shares
                nsharessell = -sells[countersell].shares
                nshares = min(nsharesbuy,nsharessell)
                buys[counterbuy].shares -= nshares
                sells[countersell].shares += nshares
                gains += nshares*(sells[countersell].price - buys[counterbuy].price)
                if nsharesbuy > nsharessell:
                    countersell += 1
                else:
                    counterbuy += 1
        print(gains)
        return gains            
    