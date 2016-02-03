import Queue
import sys
import threading

def read_output(pipe, funcs):
    for line in iter(pipe.readline, ''):
        for func in funcs:
            error = func(line)
            # time.sleep(1)
    pipe.close()

def write_output(get):
    for line in iter(get, None):
        sys.stdout.write(line)

def read_write_process(process):
    q = Queue.Queue()
    err = []
    tout = threading.Thread(target=read_output, args=(process.stdout, [q.put]))
    terr = threading.Thread(target=read_output, args=(process.stderr, [q.put, err.append]))
    twrite = threading.Thread(target=write_output, args=(q.get,))
    for t in (tout, terr, twrite): # start all three processes
        t.daemon = True
        t.start()
    process.wait()
    for t in (tout, terr):
        t.join()
    q.put(None)
    return err
            
def convert_arg_to_logical(arg):    
    try:
        return arg.capitalize() == 'True'
    except AttributeError: # logical
        return arg

class SubprocessError(Exception):
    def __init__(self,activity,error):
        self.activity = activity
        self.error = error
        
    def __str__(self):
        print('')
        print('Something went wrong during {0}'.format(activity))
        print('Dump of stderr: ')
        for line in self.error:
            print(line)
        
# Obsolete

def print_process(process):
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print line,
    process.wait() # wait for the subprocess to exit

# (Python 3 version)
# def print_process(process):
    # with process as p:
        # for line in p.stdout:
            # print(line, end='')
    