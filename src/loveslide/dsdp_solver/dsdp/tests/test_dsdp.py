#!/usr/bin/env python
# Created by: Zhisu Zhu, 01/06/2015
# Test can be run by "nosetests -v dsdp" under scikit-dsdp directory

"""Sanity test for scikit-dsdp module"""

from pydsdp.dsdp5 import *
from numpy import matrix
import os

tol=0.000001
target = dict(zip(["truss1", "control1", "vibra1", "mcp100"], [8.999996,-17.784626,-40.819012,-226.1573]))

def test_dsdp():
    A = matrix([
        [10, 4, 4, 0],
        [0, 0, 0, -8],
        [0, -8, -8, -2]])
    b = matrix ([
        [48] ,
        [-8],
        [20]])
    c = matrix ([
        [-11] ,
        [0] ,
        [0] ,
        [23]])
    K = {} # K is a dictionary for sizes of different cones
    # K['l'] = 0
    K['s'] = [2]

    OPTIONS={}
    OPTIONS['print']=0

    result = dsdp(A, b, c, K, OPTIONS)

def test_control1():
    result = dsdp_readsdpa(os.path.dirname(os.path.abspath(__file__))+"/examples/control1.dat-s","")
    assert(abs(result[2][1]-target["control1"])<tol)

def test_truss1():
    result = dsdp_readsdpa(os.path.dirname(os.path.abspath(__file__))+"/examples/truss1.dat-s","")
    assert(abs(result[2][1]-target["truss1"])<tol)

def test_vibra1():
    result = dsdp_readsdpa(os.path.dirname(os.path.abspath(__file__))+"/examples/vibra1.dat-s","")
    assert(abs(result[2][1]-target["vibra1"])<tol)

def test_mcp100():
    result = dsdp_readsdpa(os.path.dirname(os.path.abspath(__file__))+"/examples/mcp100.dat-s","")
    assert(abs(result[2][1]-target["mcp100"])<tol*100)

if __name__ == '__main__':
    test_dsdp()
    test_control1()
    test_truss1()
    test_vibra1()
    test_mcp100()
