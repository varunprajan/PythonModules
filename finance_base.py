import numpy as np
import pickle
import os
import pandas as pd
import datetime

def win(val):
    return np.sign(val)
    
def graduated_tax(income,schedule):
    tax = 0
    alreadytaxed = 0
    for threshold, rate in schedule:
        tax += (min(income,threshold) - alreadytaxed)*rate
        if income > threshold:
            alreadytaxed = threshold
        else:
            return tax
            
class Transaction(object):
    def __init__(self,time,assetname,shares,price):
        self.time = time
        self.assetname = assetname
        self.shares = shares
        self.price = price
        
    def __repr__(self):
        return '(Time: {}, Asset: {}, Shares: {}, Price: {:.2f})'.format(self.time,self.assetname,self.shares,self.price)

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
    def __init__(self,data,name,adjclosekey,datekey='Date',pickler=None,startdate=None,enddate=None):
        self.date = pd.to_datetime(data[datekey])
        if startdate is not None:
            idxstart = self.idx_closest_date(startdate)
        if enddate is not None:
            idxend = self.idx_closest_date(enddate)
        if startdate is not None and enddate is not None:
            self.data = data[idxstart:idxend]
        elif startdate is not None:
            self.data = data[idxstart:]
        elif enddate is not None:
            self.data = data[:idxend]
        else:
            self.data = data
        self.name = name
        self.date = pd.to_datetime(self.data[datekey])
        self.adjclose = self.data[adjclosekey]
        self.ratio = self.adjclose[1:]/self.adjclose[:-1]
        
    # read   
    @classmethod
    def from_quandl_file(cls,filepath,name,adjclosekey,reverse=False,**kwargs):
        dataframe = pd.read_csv(filepath)
        if reverse:
            dataframe = dataframe.iloc[::-1]
        return cls(data=dataframe.to_records(),name=name,adjclosekey=adjclosekey,**kwargs) 
        
    # dates        
    def idx_closest_date(self,date):
        datenew = pd.to_datetime(date)
        return self.date.searchsorted(datenew) # assumes dates are ordered
    
    # load/save
    def save(self):
        self.pickler.save(self)

    # computations over data set  
    def yield_fun(self,func,*args):
        for i, _ in enumerate(self.data):
            res = func(i,*args)
            yield res if res is not None else np.nan
            
    def plot_data(self,ynorm=None,DAYFAC=365):
        datestart = pd.to_datetime(datetime.datetime.now())
        datediff = self.date - datestart
        y = self.adjclose
        x = datediff.days/DAYFAC
        if ynorm is not None:
            y /= ynorm
        return np.column_stack((x,y))