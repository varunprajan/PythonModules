# notes:
# 1) Open orders
# 2) Can't buy fractional shares
# 3) Keep track of cash

class BalancedPortfolio(object):
    def __init__(self,name,fdesired,cash,fdriftmax=0.05,Lmin=1.4,Lmax=1.6,Ltarget=1.5):
        s
        self.name = name
        self.assets = {}
        self.fdriftmax = fdriftmax
        self.set_leverage_limits(Lmin,Lmax,Ltarget,reset=False)
        self.set_fdesired(fdesired,reset=False)
    
    # public methods
    def update(self):
        self._check_and_correct_leverage()
        self._check_and_rebalance_assets()

    def set_fdesired(self,fdesired,reset=True):
        fdesired = {assetname: Decimal(f) for assetname, f in fdesired.items()}
        ftotal = sum(fdesired.values())
        if abs(ftotal - 1) > TOL:
            raise ValueError('Desired fs of portfolio do not add up to 1')
        self.fdesired = fdesired
        self._add_and_delete_assets()
        if reset:
            self._check_and_rebalance_assets()
    
    # leverage    
    def _check_L_limits(self,Lmin,Lmax,Ltarget):
        if Lmin < 1:
            raise ValueError('All leverage limits must be at least 1'.format(attr))
        if not (Lmin <= Ltarget <= Lmax):
            raise ValueError('Lmin must be less than or equal to Ltarget; Ltarget must be less than or equal to Lmax')
    
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