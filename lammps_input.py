import myio

class LammpsLine(object):
    def __init__(self,name,arglist,idxid):
        self.name = name
        if idxid is None:
            self.id = True
        else:
            self.id = arglist[idxid]
        self.arglist = arglist        
        
    def __eq__(self,other):
        return (self.name == other.name) and (self.id == other.id)
        
    def __str__(self):
        argliststr = ' '.join(self.arglist)
        return '{} {}'.format(self.name,argliststr)

class VariableLine(LammpsLine):
    def __init__(self,arglist):
        super().__init__('variable',arglist,0) # variable name
    
class FixLine(LammpsLine):
    def __init__(self,arglist):
        super().__init__('fix',arglist,2) # fix style
        
class CommandLine(LammpsLine):
    def __init__(self,name,arglist):
        super().__init__(name,arglist,None)
        
def replace_lammps_input(filename,replacements):
    with open(filename,'r') as f:
        for line in f:
            leading = myio.get_leading_chars(line)
            line = line.strip()
            line = line.replace('\t',' ')
            if not line: # blank line
                yield '', ''
            elif line.startswith('#'): # comment
                yield leading, line
            else:
                first, *last = line.split(' ')
                print(repr(first))
                if first == 'variable':
                    oldobj = VariableLine(last)
                elif first == 'fix':
                    oldobj = FixLine(last)
                else:
                    oldobj = CommandLine(first,last)
                yield leading, replace_obj(oldobj,replacements)
    
def write_replaced_lammps_input(oldfilename,newfilename,replacements):
    with open(newfilename,'w') as f:
        for leading, obj in replace_lammps_input(oldfilename,replacements):
            f.write('{}{}\n'.format(leading,str(obj)))
    
def replace_obj(oldobj,replacements):
    for replaceobj in replacements:
        if replaceobj == oldobj:
            return replaceobj
    return oldobj # if we've gotten this far, there was no match
        
        