import finance_base as fbase

class SimplePortfolio(object):
    def __init__(self,cash,r,Ltarget,capitalgainsrate=0.15,marginrate=0.015,commission=0.005):
        self.marginrate = marginrate
        self.capitalgainsrate = capitalgainsrate
        self.commission = commission
        self.startingequity = cash
        self.time = 0
        self.shares = 0
        self.r = r
        self.Ltarget = Ltarget
        self.cash = cash
        self.shareprice = 100
        self.transactionhistory = []
        self.purchase_shares(cash)
        self.borrow_money()
    
    @property
    def equity(self):
        return self.portfolio_value + self.cash
    
    @property
    def leverage(self):
        if self.cash < 0:
            return self.portfolio_value/self.equity
        else:
            return 1
    
    @property
    def portfolio_value(self):
        return self.shareprice*self.shares
        
    def borrow_money(self):
        deltaborrow = (self.Ltarget - self.leverage)*self.equity
        self.purchase_shares(deltaborrow)
        
    def purchase_shares(self,cash):
        nshares = int(round(cash/self.shareprice))
        value = nshares*self.shareprice
        self.cash -= value
        self.shares += nshares
        self.cash -= abs(nshares)*self.commission
        self.transactionhistory.append(fbase.Transaction(self.time,'myasset',nshares,self.shareprice))
        
    def advance_one_year(self):
        self.shareprice *= 1 + self.r
        self.time += 1
        self.apply_margin()
        self.borrow_money()
            
    def apply_margin(self):
        schedule = [(1e5,self.marginrate),(1e6,self.marginrate - 0.005),(1e10,self.marginrate - 0.01)]
        self.cash -= graduated_tax(-self.cash,schedule)
        
    def liquidate(self):
        self.purchase_shares(-self.portfolio_value)
        self.apply_capital_gains()        
        
    def apply_capital_gains(self):
        gains = self.naive_capital_gains()
        print(gains)
        self.cash -= self.capitalgainsrate*gains
        
    def naive_capital_gains(self):
        gains = 0
        temp = copy.deepcopy(self.transactionhistory)
        buys = [transaction for transaction in temp if transaction.shares > 0]
        buys = sorted(buys, key = lambda x: x.time)
        sells = [transaction for transaction in temp if transaction.shares < 0]
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
        return gains
        
    def annualized_r(self):
        return (self.equity/self.startingequity)**(1/self.time) - 1
        
    def view(self):
        print('Equity: ${:.2f}'.format(self.equity))
        print('Leverage: {:.3f}'.format(self.leverage))
        print('Portfolio value: ${:.2f}'.format(self.portfolio_value))