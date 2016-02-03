import pickle
import scipy.io as spio
import operator as op
import os
import tempfile
import numpy as np
from numpy import warnings
import subprocess

def get_leading_chars(line):
    idx = len(line)-len(line.lstrip())
    return line[:idx]
                
def np_array(f): # read generator, return nparray (None if generator is empty)
    res = np.loadtxt(f)
    if res.size:
        return res
    else:
        return None
                
def yield_last_item(f): # get last element in generator (None if generator is empty)
    try:
        for el in f:
            pass
        return el
    except UnboundLocalError:
        return None

def yield_tagged_lines(f,tag):
    for line in f:
        if line.startswith(tag):
            yield line
        
def yield_lines_between(f,starttag,endtag):
    read = False
    for line in f:
        line = line.strip()
        if line.startswith(starttag):
            read = True
        elif line.startswith(endtag):
            return
        elif read:
            yield line
                
def yield_until(f,endtag):
    for line in f:
        line = line.strip()
        if line.startswith(endtag):
            return
        else:
            yield line
        
def my_read_array(f):
    with warnings.catch_warnings(): # prevent numpy from outputting empty array warning
        warnings.simplefilter("ignore")
        res = np.loadtxt(f)
    if np.array_equal(np.round(res),res):
        return res.astype(int)
    else:
        return res

def my_write_array(f,array,fmt=None):
    try: # reformat single column arrays
        array.shape[1]
    except IndexError:
        array = np.reshape(array,(array.shape[0],1))
    ncol = array.shape[1]
    if fmt is None:
        fmt = ['{0} ']*ncol
    for line in array.tolist():
        for num, fmtnum in zip(line, fmt):
            try:
                f.write(fmtnum.format(num))
            except ValueError:
                f.write(fmtnum.format(int(num)))
        f.write('\n')
        
def coerce_string(s):        
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
            
def yield_no_comments(f,comment='#'):
    for line in f:
        line = line.strip()
        if line and not line.startswith(comment):
            yield line
                
def findKey(line,keystart,keyend):
    indexstart = str.find(line,keystart)
    line = line[indexstart+len(keystart):]
    indexend = str.rfind(line,keyend)
    return line[:indexend]
            
def getFilePrefix(filepath):
    return getFileAttr(filepath)[0]
    
def getFileExt(filepath):
    return getFileAttr(filepath)[1]
    
def getFileAttr(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)

def findCharIndices(line,char):
    return [i for i, letter in enumerate(line) if letter == char]
    
def getMatlabObject(filename):
    return spio.loadmat(filename)

def inPlaceReplace(dirpath,filename,namesold,namesnew,newline=''):
    tempfilename = filename + '_temp_temp'
    with open(dirpath+tempfilename,'wt',newline=newline) as ftemp:
        with open(dirpath+filename,'rt',newline=newline) as f:
            for line in f:
                linenew = line
                for str1, str2 in zip(namesold,namesnew):
                    linenew = linenew.replace(str1,str2)
                ftemp.write(linenew)
        os.remove(dirpath+filename)
    os.rename(dirpath+tempfilename,dirpath+filename)
    
def copyReplaceFile(stroldlist,strnewlist,filenameold,filenamenew=None,subdir=''):
    # copy file, replacing strold with strnew in both file and filename
    # if filenew is None, rename new file to filenew
    if filenamenew is None:
        filenamenew = multipleReplace(filenameold,stroldlist,strnewlist)
    if filenamenew != filenameold: # don't overwrite file!
        with open(subdir + filenameold, 'r') as f:
            with open(subdir + filenamenew, 'w') as f2:
                for line in f:
                    f2.write(multipleReplace(line,stroldlist,strnewlist))
                
def multipleReplace(line,stroldlist,strnewlist):
    for strold, strnew in zip(stroldlist,strnewlist):
        line = line.replace(strold,strnew)
    return line
    
def bashBatchRun(bashfile,jobfile,filepref,searchstring='FILE_PREF='):
    with open(jobfile, 'r') as jobs:
        for line in jobs:
            job = line.rstrip()
            with open(bashfile, 'r') as bash:
                with open('temp', 'w') as temp:
                    for line in bash:
                        if line.startswith(searchstring):
                            linenew = searchstring + filepref + job
                            temp.write(linenew + '\n')
                        else:
                            temp.write(line)
            os.remove(bashfile)
            os.rename('temp',bashfile)
            subprocess.call('bash ' + bashfile,shell=True)
    