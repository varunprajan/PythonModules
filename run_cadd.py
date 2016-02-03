from subprocess import Popen, PIPE
import paths
import os
import shutil
import misc_cadd as mc
import sys
import cadd_io as cdio

dumpsuffix = '.dump'
restartsuffix = '.restart'
simfilesuffix = '_nodes'
mainfile = 'main'
lammpspref = 'lammps_'

def run_cadd(exefile,lammpsoutput=False,cleanup=True,simname=None,inputpath=paths.inputpath,runpath=paths.runpath,outputpath=paths.outputpath):
    if simname is None:
        simname = get_simname(inputpath)
    make_main_file(simname,inputpath)
    copy_input_files(inputpath,runpath)
    bashcommand = './{0}'.format(exefile)
    process = Popen([bashcommand],stdout=PIPE,stderr=PIPE,cwd=runpath)
    err = mc.read_write_process(process)
    if err:
        raise mc.SubprocessError('Running the executable',err)
    copy_output_files(outputpath,runpath)
    if mc.convert_arg_to_logical(cleanup):
        clean_up_files(runpath,exefile)
    if mc.convert_arg_to_logical(lammpsoutput):
        lammps_output(outputpath)

def get_simname(inputpath,suffix=simfilesuffix):
    # assumes only file with suffix is a file associated with simulation
    for filename in os.listdir(inputpath):
        if filename.endswith(suffix):
            return filename.split(suffix)[0]
            
def make_main_file(simname,inputpath,mainfile=mainfile):
    with open(inputpath+mainfile,'w') as f:
        f.write('{0}\n'.format(simname))
        
def copy_input_files(inputpath,runpath):
    for filename in os.listdir(inputpath):
        shutil.copy(inputpath+filename,runpath+filename)
        
def copy_output_files(outputpath,runpath,suffixes=(dumpsuffix,restartsuffix)):
    for filename in os.listdir(runpath):
        if filename.endswith(suffixes):
            shutil.copy(runpath+filename,outputpath+filename)
        
def clean_up_files(runpath,exefile):
    # delete all files except exefile
    for filename in os.listdir(runpath):
        if filename != exefile:
            os.remove(runpath+filename)
            
def restart_to_input(outputpath,inputpath,suffix=restartsuffix):
    for filename in os.listdir(outputpath):
        if filename.endswith(suffix):
            fileprefix, _ = get_prefix_incr(filename)
            shutil.copy(outputpath+filename,inputpath+fileprefix)
            
def lammps_output(outputpath,suffix=dumpsuffix,prefix=lammpspref):
    for filename in os.listdir(outputpath):
        if filename.endswith(suffix):
            _, increment = get_prefix_incr(filename)
            datadict = cdio.readInput(outputpath+filename)
            cdio.writeLammpsDump(datadict,outputpath+prefix+filename,increment)
            
def get_prefix_incr(filename):
    lastdotindex = filename.rfind('.')
    fileprefix = filename[:lastdotindex]
    lastdotindex = fileprefix.rfind('.')
    increment = int(fileprefix[lastdotindex+1:])  
    fileprefix = filename[:lastdotindex]
    return fileprefix, increment   
            
run_cadd(*sys.argv[1:])
    