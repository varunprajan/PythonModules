from decimal import Decimal
import pandas as pd
import numpy as np

TOL = 1e-10
TOLCASH = Decimal(0.00001)
ZERO = Decimal(0)
LMARGIN = 2.0

class Simulation(object):
    def __init__(self,startdate,enddate,timeseries):
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate)
        self.timeseries = timeseries
        self.init_idx_dates()
        nseries = len(self.timeseries.keys())
        self.performance = np.empty((self.idxdiff,nseries+1))
        
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
    
    def run_simulation(self,portfolio):
        assetnames = list(portfolio.assets.keys())
        for counter in range(self.idxdiff):
            for i, name in enumerate(assetnames):
                idxcurr = self.idx[name][0] + counter
                priceratio = Decimal(self.timeseries[name].ratio[idxcurr])
                portfolio.assets[name] *= priceratio
                self.performance[counter,i] = portfolio.assets[name]
            self.performance[counter,-1] = portfolio.cash
            portfolio.update()
                
    def asset_curr(self,counter):
        return np.sum(self.performance[counter,:])
    
    def time_plot(self):
        return np.arange(self.idxdiff)
    
    def individual_performance_plot(self):
        xplot = self.time_plot()
        return [np.column_stack((xplot,yplot)) for yplot in self.performance.T]
        
    def total_performance_plot(self):
        xplot = self.time_plot()
        yplot = np.sum(self.performance,axis=1)
        return [np.column_stack((xplot,yplot))]
        
    def yearly_return(self,DAYFAC=365):
        final = self.asset_curr(-1)
        initial = self.asset_curr(1)
        timediff = self.enddate - self.startdate
        nyears = timediff.days/DAYFAC
        return (final/initial)**(1/nyears) - 1

class BalancedPortfolio(object):
    def __init__(self,name,fdesired,cash,fdriftmax,Lmin,Lmax,Ltarget):
        self.name = name
        self.assets = {}
        self.cash = ZERO
        self.fdriftmax = fdriftmax
        self.set_leverage_limits(Lmin,Lmax,Ltarget,reset=False)
        self.assign_new_allocation(fdesired,reset=False)
        self.deposit_or_withdraw_cash(cash)
        
    @classmethod
    def test_portfolio(cls,fstocks=None,fbonds=None,fgold=None,name='try',cash=10000,fdriftmax=0.05,Lmin=1.45,Lmax=1.55,Ltarget=1.5):
        fdesired = {}
        if fstocks is not None:
            fdesired['Stocks'] = fstocks
        if fbonds is not None:
            fdesired['Bonds'] = fbonds
        if fgold is not None:
            fdesired['Gold'] = fgold
        return cls(name,fdesired,cash,fdriftmax,Lmin,Lmax,Ltarget)
    
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

    def deposit_or_withdraw_cash(self,amount):
        self._check_equity(amount)
        self._modify_cash(amount)
        self._order_allocated_assets(amount)
        self._check_and_correct_leverage()
    
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
                    yield '\t{}: ${:.2f}'.format(assetname,amount)
                yield 'Cash: ${:.2f}'.format(self.cash)
        return '\n'.join(string for string in yield_str())

    # basic properties
    @property
    def _leverage_ratio(self):
        cashfudge = self.cash if self.cash > 0 else 0
        return (self._portfolio_value + cashfudge)/self._equity
    
    @property
    def _portfolio_value(self):
        return sum(self.assets.values())

    @property
    def _equity(self):
        return self._portfolio_value + self.cash
    
    # deposits, withdrawals, borrowing    
    def _modify_cash(self,delta):
        self.cash += delta
       
    def _check_equity(self,delta):
        if self._equity + delta < ZERO:
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
            currentholding = self.assets[assetname]
            self._order_asset(assetname,-currentholding)
            del self.assets[assetname]
        for assetname in addedassets:
            self.assets[assetname] = ZERO
    
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
        return self.assets[assetname]/portfoliovalue
    
    # sell, buy assets    
    def _order_allocated_assets_using_cash(self):
        if self.cash > ZERO:
            self._order_allocated_assets(self.cash)
    
    def _order_allocated_assets(self,total):
        orders = self._allocated_orders(total)
        self._order_assets(orders)
    
    def _allocated_orders(self,total):
        return [(assetname, total*fdesired) for (assetname, fdesired) in self.fdesired.items()]
    
    def _order_assets(self,orders):
        for assetname, amount in orders:
            self._order_asset(assetname,amount)           
    
    def _order_asset(self,assetname,amount):
        # print('Ordering asset {} in amount ${}'.format(assetname,amount))
        self._modify_cash(-amount)
        self._modify_asset(assetname,amount)
        
    def _modify_asset(self,assetname,delta):
        """Change asset amount, enforcing the condition that shorting is not allowed"""
        currentamount = self.assets[assetname]
        newamount = currentamount + delta
        if newamount < ZERO:
            raise ValueError('Current holding in {} is {:.2f}, while requested change is {:.2f}'.format(assetname,currentamount,delta))
        self.assets[assetname] = newamount
    