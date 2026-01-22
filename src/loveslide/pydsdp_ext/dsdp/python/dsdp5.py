#!/usr/bin/env python

__all__ = ['dsdp', 'dsdp_readsdpa']

from pydsdp.convert import *
from numpy import *
from pydsdp.pydsdp5 import pyreadsdpa
from os import remove, path
from tempfile import NamedTemporaryFile

def dsdp(A, b, c, K, OPTIONS={}):
    tempdataF = NamedTemporaryFile(delete=False)
    data_filename=tempdataF.name
    tempdataF.close()
    sedumi2sdpa(data_filename, A, b, c, K)

    
    # tempoptionsF = None
    options_filename = ""
    if len(OPTIONS)>0:
        tempoptionsF = NamedTemporaryFile(delete=False)
        options_filename = tempoptionsF.name
        tempoptionsF.close()
        write_options_file(options_filename, OPTIONS)

        
    # solve the problem by dsdp5
    [y,X,STATS] = dsdp_readsdpa(data_filename,options_filename)

    if path.isfile(data_filename): 
        remove(data_filename)
    if path.isfile(options_filename): 
        remove(options_filename)

    if not 'l' in K: K['l']=0
    if not 's' in K: K['s']=()
    Xout = []
    if K['l']>0: Xout.append(X[0:K['l']])
    
    index = K['l'];
    if 's' in K:
        for d in K['s']:
            Xout.append(matrix(reshape(array(X[index:index+d*d]), [d,d])))
            index = index+d*d

    if (STATS[0]==1):
        STATS[0] = "PDFeasible"
    elif (STATS[0]==3):
        STATS[0] = "Unbounded"
    elif (STATS[0]==4):
        STATS[0] = "InFeasible"

    STATSout = {}
    # Assign the name to STATS output, should be consistent with Rreadsdpa.c 
    if (len(STATS)>3):
        STATSout=dict( zip(["stype", "dobj","pobj","r","mu","pstep","dstep","pnorm"],STATS) )
    else:
        STATSout=dict( zip(["stype", "dobj","pobj"], STATS) )

    return dict(zip(['y', 'X','STATS'],[y,Xout,STATSout]))
    

def dsdp_readsdpa(data_filename,options_filename):
    # solve the problem by pyreadsdpa
    # result = [y,X,STATS]
    result = []
    if ( path.isfile(data_filename) and (options_filename=="" or path.isfile(options_filename)) ):
        result = pyreadsdpa(data_filename,options_filename)
    return result


def write_options_file(filename, OPTIONS):
    # write OPTIONS to file
    with open(filename, "a") as f:
        for option in OPTIONS.keys():
            f.write("-" + option + " " +str(OPTIONS[option]) + "\n")
    f.close()
