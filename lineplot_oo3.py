import matplotlib as mpl
import copy
import my_plot as myplot

def my_quick_plot(data,**kwargs):
    plotdata = LinePlotData.from_dataset(data)
    fig = myplot.my_plot(plotdata)
    myplot.pretty_figure(fig,**kwargs)

def get_setter_fun(attribute):
    return 'set_' + attribute
    
def duplicated_list(stringlist,ndup):
    '''E.g. duplicated_list(['b','k','r'],2)
    yields ['b','b','k','k','r','r']'''
    return [el for el in stringlist for _ in range(ndup)]

class Style(dict):
    def __init__(self,color=None,linestyle=None,marker=None,markersize=None,linewidth=None):
        if color is None:
            color = LineAttributes.color_default()
        if linestyle is None:
            linestyle = LineAttributes.linestyle_default()
        if marker is None:
            marker = LineAttributes.marker_default()
        if markersize is None:
            markersize = LineAttributes.markersize_default()
        if linewidth is None:
            linewidth = LineAttributes.linewidth_default()
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.linewidth = linewidth
    
    def alternate_between(self,attr1,attr2,n):
        getattr(self,attr1).select_first_n(n)
        getattr(self,attr2).duplicate_with_n(n)

    @classmethod
    def alternating_1(cls,n):
        self = cls()
        self.alternate_between('linestyle','color',n)
        return self
        
    @classmethod
    def alternating_2(cls,n):
        self = cls()
        self.alternate_between('color','linestyle',n)
        return self
    
class LineAttributes(object):
    def __init__(self,attrlist):
        self.attrlist = attrlist
        
    def get_num_items(self):
        return len(self.attrlist)
        
    def select_first_n(self,n):
        self.attrlist = self.attrlist[:n]
        
    def duplicate_with_n(self,n):
        self.attrlist = duplicated_list(self.attrlist,n)
        
    @classmethod
    def color_default(cls):
        return cls(['k','r','b','g','m','c','y'])
    
    @classmethod
    def linestyle_default(cls):
        return cls(['-','--','-.',':'])
        
    @classmethod
    def marker_default(cls):
        return cls([''])
        
    @classmethod
    def markersize_default(cls):
        return cls([10])
    
    @classmethod
    def linewidth_default(cls):
        return cls([2.5])
        
class LinePlotData(object):    
    def __init__(self,linelist=None,scale=(1,1),style=None):
        if linelist is None:
            linelist = []
        self.linelist = linelist
        self.scale = scale
        if style is None:
            style = Style()
        self.style = style
        
    @classmethod
    def from_dataset(cls,dataset,style=None):
        obj = cls(style=style)
        obj.add_many_datasets(dataset)
        obj.set_styles()
        return obj
        
    def add_line(self,newline):
        self.linelist.append(newline)
        
    def get_num_lines(self):
        return len(self.linelist)
        
    def add_many_datasets(self,dataall):
        for data in dataall:
            try:
                xdata, ydata = data
            except ValueError:
                xdata, ydata = data[:,0], data[:,1]
            newdata = mpl.lines.Line2D(xdata,ydata)
            self.add_line(newdata)
            
    def add_to_axes(self,ax):
        xscale, yscale = self.scale
        for line in self.linelist:
            linenew = copy.copy(line) # don't want to overwrite data
            xdata, ydata = linenew.get_data()
            xdata, ydata = xdata*xscale, ydata*yscale
            linenew.set_data(xdata,ydata)
            ax.add_line(linenew)

    def set_styles(self):
        for attribute, lineattributes in self.style.__dict__.items():
            attrlist = lineattributes.attrlist
            setterattr = get_setter_fun(attribute)
            nitems = lineattributes.get_num_items()
            for i, line in enumerate(self.linelist):
                idx = i%nitems # cycle if end of list is reached
                getattr(line,setterattr)(attrlist[idx]) # call setterfun, changing value to attrlist[idx]
