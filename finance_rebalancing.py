from decimal import Decimal
import pandas as pd
import numpy as np

TOL = 1e-10
TOLCASH = Decimal(0.00001)
ZERO = Decimal(0)

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
            self.performance[counter,-1] = portfolio.borrowed
            portfolio.update()
                
    def asset_curr(self,counter):
        return np.sum(self.performance[counter,:-1]) - self.performance[counter,-1]
    
    def time_plot(self):
        return np.arange(self.idxdiff)
    
    def individual_performance_plot(self):
        xplot = self.time_plot()
        return [np.column_stack((xplot,yplot)) for yplot in self.performance.T]
        
    def total_performance_plot(self):
        xplot = self.time_plot()
        yplot = np.sum(self.performance[:,:-1],axis=1) - self.performance[:,-1]
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
        self.borrowed = ZERO
        self.fdriftmax = fdriftmax
        self.tradevolume = ZERO
        self.set_leverage_limits(Lmin,Lmax,Ltarget,reset=False)
        self.set_fdesired(fdesired,reset=False)
        self.deposit_cash(cash)
        
    @classmethod
    def test_portfolio(cls,fstocks=None,fbonds=None,fgold=None,name='try',cash=1000,fdriftmax=0.05,Lmin=1.4,Lmax=1.6,Ltarget=1.5):
        fdesired = {}
        if fstocks is not None:
            fdesired['Stocks'] = fstocks
        if fbonds is not None:
            fdesired['Bonds'] = fbonds
        if fgold is not None:
            fdesired['Gold'] = fgold
        return cls(name,fdesired,cash,fdriftmax,Lmin,Lmax,Ltarget)
        
    def _add_and_delete_assets(self):
        oldassets = set(self.assets.keys())
        newassets = set(self.fdesired.keys())
        addedassets = newassets.difference(oldassets)
        deletedassets = oldassets.difference(newassets)
        for assetname in deletedassets:
            currentholding = self.assets[assetname]
            self._sell_asset(assetname,currentholding)
            del self.assets[assetname]
        for assetname in addedassets:
            self.assets[assetname] = ZERO
        if deletedassets: # we may have available cash from selling the deleted assets
            self._buy_assets()
    
    # public methods
    def update(self):
        self._check_and_correct_leverage()
        self._check_and_rebalance_assets()
    
    def view(self,verbose=True):
        print(self._stringify(verbose))

    def set_fdesired(self,fdesired,reset=True):
        fdesired = {assetname: Decimal(f) for assetname, f in fdesired.items()}
        ftotal = sum(fdesired.values())
        if abs(ftotal - 1) > TOL:
            raise ValueError('Desired fs of portfolio do not add up to 1')
        self.fdesired = fdesired
        self._add_and_delete_assets()
        if reset:
            self._check_and_rebalance_assets()
            
    def set_leverage_limits(self,Lmin,Lmax,Ltarget,reset=True):
        self._check_L_limits(Lmin,Lmax,Ltarget)
        self.Lmin = Decimal(Lmin)
        self.Lmax = Decimal(Lmax)
        self.Ltarget = Decimal(Ltarget)
        if reset:
            self.check_and_correct_leverage()

    def deposit_cash(self,amount):
        self._check_nonnegative_amount(amount,'Deposit')
        self._modify_cash(amount)
        self._buy_assets()
        self._check_and_correct_leverage()
    
    def withdraw_cash(self,amount):
        self._check_nonnegative_amount(amount,'Withdrawal')
        self._check_equity(amount)
        self._sell_assets(amount)
        self._modify_cash(-amount)
        # actually withdraw cash here
        self._check_and_correct_leverage()
    
    # view portfolio
    def __str__(self):
        return self._stringify(verbose=False)
    
    def _stringify(self,verbose):
        def yield_str():
            yield 'Portfolio {}'.format(self.name)
            yield 'Total assets: ${:.2f}'.format(self._total_assets)
            if verbose:
                yield 'Asset allocation:'
                for assetname, amount in self.assets.items():
                    yield '\t{}: ${:.2f}'.format(assetname,amount)
                yield 'Borrowed cash: ${:.2f}'.format(self.borrowed)
                yield 'Cash: ${:.2f}'.format(self.cash)
                yield 'Leverage ratio: {:.3f}'.format(self._leverage_ratio)
        return '\n'.join(string for string in yield_str())

    # basic properties, initialization
    @property
    def _total_assets(self):
        return self._total_assets_without_cash + self.cash
    
    @property
    def _total_assets_without_cash(self):
        return sum(self.assets.values())

    @property
    def _equity(self):
        return self._total_assets - self.borrowed
    
    # deposits, withdrawals, borrowing    
    def _modify_cash(self,delta):
        if self.cash + delta < ZERO:
            raise ValueError('Current cash is only {}, while requested cash was {}'.format(self.cash,-delta))
        self.cash += delta
       
    def _check_equity(self,withdrawalamount):
        equity = self._equity
        if equity - withdrawalamount < TOLCASH:
            raise ValueError('Attempting to withdraw more money (${}) than current equity (${})'.format(withdrawalamount,equity))
        
    def _modify_borrowed_cash(self,delta):
        if self.borrowed + delta < ZERO:
            raise ValueError('Current borrowed cash is {}, while requested change is {}'.format(self.borrowed,delta))
        self.borrowed += delta 
    
    def _check_nonnegative_amount(self,amount,action):
        if amount < TOLCASH:
            raise ValueError('{} amount (${}) is too small or negative'.format(action,amount))       
        
    def _borrow_or_repay_cash(self,delta):
        if delta < -TOLCASH:
            self._repay_borrowed_cash(-delta)
        elif delta > TOLCASH:
            self._borrow_cash(delta)
    
    def _borrow_cash(self,amount):
        self._modify_cash(amount)
        self._modify_borrowed_cash(amount)
        # actual order to borrow would need to be inserted
        print('Borrowing cash: ${:.2f}'.format(amount))
        
    def _repay_borrowed_cash(self,amount):
        self._modify_cash(-amount)
        self._modify_borrowed_cash(-amount)
        # actual order to repay would need to be inserted
        print('Repaying borrowed cash: ${:.2f}'.format(amount)) 
    
    # leverage    
    def _check_L_limits(self,Lmin,Lmax,Ltarget):
        if Lmin < 1:
            raise ValueError('All leverage limits must be at least 1'.format(attr))
        if not (Lmin <= Ltarget <= Lmax):
            raise ValueError('Lmin must be less than or equal to Ltarget; Ltarget must be less than or equal to Lmax')
    
    @property
    def _leverage_ratio(self):
        return self._total_assets/self._equity
    
    def _check_and_correct_leverage(self):
        Lcurr = self._leverage_ratio
        if self._bad_leverage(Lcurr):
            Lnew = self._target_leverage(Lcurr)
            deltaborrow = self._amount_to_borrow(Lcurr,Lnew)
            if deltaborrow < ZERO: # need to pay down borrowed money by selling assets
                self._sell_assets(-deltaborrow)
                self._borrow_or_repay_cash(deltaborrow)
            else: # need to borrow more money to buy assets
                self._borrow_or_repay_cash(deltaborrow)
                self._buy_assets() # buy using total available cash
    
    def _bad_leverage(self,Lcurr):
        return not (self.Lmin <= Lcurr <= self.Lmax)
        
    def _amount_to_borrow(self,Lcurr,Lnew):
        return (Lnew-Lcurr)/Lcurr*self._total_assets
        
    def _target_leverage(self,Lcurr):
        return self.Ltarget       

    # drift/rebalancing
    def _check_and_rebalance_assets(self):
        if self._f_drift > self.fdriftmax:
            self._rebalance_assets()        
        
    def _rebalance_assets(self):
        def yield_trade():
            cashlessassets = self._total_assets_without_cash
            for assetname, fdesired in self.fdesired.items():
                factual = self._f_actual(assetname,cashlessassets)
                amount = cashlessassets*(fdesired - factual)
                yield assetname, amount
        alltradessorted = sorted(yield_trade(),key=lambda x: x[1]) # need to sell assets before buying others
        lastasset, amountlast = alltradessorted[-1]
        lasttradenew = lastasset, amountlast - TOLCASH # fudge slightly to buy less than what is needed
        alltradessorted[-1] = lasttradenew 
        for assetname, amount in alltradessorted:
            self._buy_or_sell_asset(assetname,amount)
            
    @property
    def _f_drift(self):
        cashlessassets = self._total_assets_without_cash
        def yield_drift():
            for assetname, fdesired in self.fdesired.items():
                factual = self._f_actual(assetname,cashlessassets)
                yield abs(factual - fdesired)
        return sum(yield_drift())
    
    def _f_actual(self,assetname,cashlessassets):
        return self.assets[assetname]/cashlessassets
    
    # sell, buy assets    
    def _sell_assets(self,total):
        total += TOLCASH # fudge up by a small amount to ensure we get at least total
        for assetname, fdesired in self.fdesired.items():
            sellamount = total*fdesired
            self._sell_asset(assetname,sellamount)
    
    def _buy_assets(self,total=None):
        if total is None:
            total = self.cash
        total -= TOLCASH # fudge by a small amount to ensure we don't exceed available money
        for assetname, fdesired in self.fdesired.items():
            buyamount = total*fdesired
            self._buy_asset(assetname,buyamount)             
    
    def _buy_or_sell_asset(self,assetname,amount):
        if amount > TOLCASH:
            self._buy_asset(assetname,amount)
        elif amount < -TOLCASH:
            self._sell_asset(assetname,-amount)
    
    def _sell_asset(self,assetname,amount):
        if amount > TOLCASH:
            self._modify_cash(amount)
            self._modify_asset(assetname,-amount)
            self.tradevolume += amount
            # actual order to sell would need to be inserted
            # print('Selling asset {}: ${:.2f}'.format(assetname,amount))
    
    def _buy_asset(self,assetname,amount):
        if amount > TOLCASH:
            self._modify_cash(-amount)
            self._modify_asset(assetname,amount)
            self.tradevolume += amount
            # actual order to buy would need to be inserted
            # print('Buying asset {}: ${:.2f}'.format(assetname,amount))
        
    def _modify_asset(self,assetname,delta):
        currentamount = self.assets[assetname]
        newamount = currentamount + delta
        if newamount < ZERO:
            raise ValueError('Current holding in {} is {:.2f}, while requested change is {:.2f}'.format(assetname,currentamount,delta))
        self.assets[assetname] = newamount
    