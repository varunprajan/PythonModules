from subprocess import Popen, PIPE, STDOUT
import os
import sys
import paths
import shutil
import misc_cadd as mc

exefileold = 'foo'
exedir = 'bin/'
builddir = 'build/'
compiletypes = ('testing','debug','release')

def compile_cadd_and_move(exefile,clean=False,compiletype='release',compilepath=paths.compilepath,runpath=paths.runpath):
    clean = mc.convert_arg_to_logical(clean)
    compile_cadd(compiletype,compilepath,clean)
    move_executable(exefile,compilepath,runpath)

def compile_cadd(compiletype,compilepath,clean,builddir=builddir):
    buildpath = compilepath + builddir
    if compiletype not in compiletypes:
        raise ValueError('Undefined compile type')
    if clean:
        cmakecommand = 'cmake .. -DCMAKE_BUILD_TYPE={0}'.format(str.upper(compiletype))
        bashcommands = ['make distclean',cmakecommand,'make']
    else:
        bashcommands = ['make']
    for bashcommand in bashcommands:
        process = Popen(bashcommand.split(),stdout=PIPE,stderr=PIPE,cwd=buildpath)
        err = mc.read_write_process(process)
        if err:
            raise mc.SubprocessError('Compiling the executable',err)

def move_executable(exefile,compilepath,runpath,exefileold=exefileold,exedir=exedir):                
    src = compilepath + exedir + exefileold
    dst = runpath + exefile
    if not os.path.isdir(runpath):
        os.mkdir(runpath)
    shutil.move(src,dst)
    
compile_cadd_and_move(*sys.argv[1:])



