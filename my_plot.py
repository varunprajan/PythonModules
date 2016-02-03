import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# blah

def my_plot(plotdata,fig=None,fignum=1,figuresize=None,font='Arial',show=True,clear=True):
    '''Creates a figure with one subplot.
    Alternatively, clears lines from old figure and plots data.
    Can be prettified using pretty_figure.
    Requires that plotdata is an object with a method add_to_axes
    that takes an Axes instance'''
    if fig is None:
        fig = gen_figure(fignum,figuresize)
        fig.add_subplot(111)
    elif clear:
        clear_fig(fig)
    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42
    ax = fig.axes[0]
    plotdata.add_to_axes(ax) # polymorphism
    if show:
        fig.show()
    return fig
    
def clear_lines(fig):
    '''Needs some work. Purposes is to clear all plotted objects from fig'''
    for ax in fig.axes:
        ax.lines = []

def gen_figure(fignum=1,figuresize=None):
    '''Create figure with specified number and size.
    Size defaults to 8 by 6 inches (which looks nice).'''
    if figuresize is None:
        figuresize = [8,6]
    fig = plt.figure(fignum,figsize=figuresize)
    fig.clf()
    return fig
    
def pretty_figure(fig,aspect=None,xlabel=None,ylabel=None,axisbounds=None,fontsizeaxes=21,fontsizeother=18,
                  ticksize=False,borderwidth=2,tight=True,tightlayout=True,tightfac=1.08,ticksizedef=[8,2]):
    '''Prettifies a figure with labels, proper linewidths and font sizes, ticks, tight axes/layout, etc.
    Note that for the axisbounds, there are three options: direct specification (axisbounds)
    tight = True (tight axes), or tight = False (tight axes with small padding around outside).
    Also, aspect should be set to 1 for plots of physical objects'''
    ax = fig.axes[0]
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsizeaxes)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsizeaxes)
    if axisbounds is not None:
        ax.axis(axisbounds)
    else:
        ax.autoscale()
        ax.axis('tight')
        if not tight:
            xcenter = np.mean(ax.get_xlim())
            ycenter = np.mean(ax.get_ylim())
            xlimnottight = [(x-xcenter)*tightfac + xcenter for x in ax.get_xlim()]
            ylimnottight = [(y-ycenter)*tightfac + ycenter for y in ax.get_ylim()]
            ax.axis(xlimnottight + ylimnottight)
    if aspect is not None:
        ax.set_aspect(aspect)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(borderwidth)
    if ticksize is None:
        ax.tick_params(axis='both',labelsize=fontsizeother,bottom='off',top='off',left='off',right='off')
    else:
        if not ticksize:
            ticksize = ticksizedef      
        ax.tick_params(axis='both',labelsize=fontsizeother,width=ticksize[1],length=ticksize[0])
    if tightlayout:
        fig.tight_layout()
    return fig