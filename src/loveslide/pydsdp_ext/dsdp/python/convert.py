#!/usr/bin/env python
"""convert.py

Input SDPA sparse format (dat-s) file from sedumi file.

"""

__all__ = ['sedumi2sdpa']

from scipy.sparse import csc_matrix, csr_matrix
from scipy import sparse
from numpy import matrix

def sedumi2sdpa(filename, A, b, c, K):
    """Convert from Sedumi format to SDPA sparse format
    
    Arguments:
      filename: A string of output sdpa file name
      A, b, c, K: Input in sedumi format, where A,b,c are in Scipy matrices format
    """

    A = sparse.csc_matrix(A)
    b = sparse.csc_matrix(b)
    c = sparse.csc_matrix(c)
    if c.get_shape()[1]>1: c=c.transpose()

    if not sparse.isspmatrix_csc(A):
        A = A.tocsc()
    if not sparse.isspmatrix_csc(b):
        b = b.tocsc()
    if not sparse.isspmatrix_csc(c):
        c = c.tocsc()

    if not 'l' in K.keys():
        K['l'] = 0 

    fp = open(filename, "w")

    # write mDim
    mDim = A.get_shape()[0]
    fp.write(str(mDim) + "\n")

    # write nBlock
    if K['l']>0:
        fp.write(str(1+len(K['s'])) + "\n")
    else:
        fp.write(str(len(K['s'])) + "\n")

    # write blockStruct
    if K['l']>0:
        fp.write(str(-K['l']) + " ")
    fp.write(" ".join([ str(x) for x in K['s'] ]) + "\n")

    # write b
    fp.write(" ".join([ str(-b[i,0]) for i in range(b.shape[0])])  + "\n")

    # write C
    len_s = sum([x ** 2 for x in K['s'] ])
        

    matnum = 0
    blocknum = 0
    curind = 0
    # block one for linear cone
    if K['l']>0:
        blocknum += 1
        c_l = -c[0:K['l'], :]
        list_row = [str(x + 1) for x in list(c_l.nonzero()[0])]
        list_val = [x for x in list(c_l.data)]
        length = len(list_row)
        for i in range(length):
            fp.write(" ".join((str(matnum), str(blocknum), list_row[i],
                               list_row[i], str(list_val[i]))) + "\n") 
        curind = curind + K['l']

    if len_s > 0:
        offset = 0
        c_s = -c[curind :(curind + len_s), :]
        for blockSize in K['s']:
            blocknum += 1
            list_row = list(c_s[offset:(offset + blockSize * blockSize),
                                :].nonzero()[0])
            list_val = [x for x in
                        list(c_s[offset:(offset + blockSize * blockSize),
                                 :].data)]
            length = len(list_row)
            for i in range(length):
                setCol_row = (list_row[i] // blockSize) + 1
                setCol_col = (list_row[i] % blockSize) + 1
                if setCol_row <= setCol_col:
                    fp.write(" ".join(("0", str(blocknum), str(setCol_row),
                                       str(setCol_col), str(list_val[i]))) + "\n")
            offset += blockSize * blockSize  

    # write A
    blocknum = 0
    curind = 0

    if K['l'] > 0:
        blocknum += 1
        A_l = -A[:, 0:K['l']]
        list_row = [str(x + 1) for x in list(A_l.nonzero()[0])]
        list_col = [str(x + 1) for x in list(A_l.nonzero()[1])]
        list_val = [x for x in list(A_l.data)]
        length = len(list_row)
        for i in range(length):
            fp.write(" ".join((list_row[i], str(blocknum), list_col[i],
                               list_col[i], str(list_val[i]))) + "\n")
        curind = curind + K['l']



    if len_s > 0:
        offset = 0
        A_s = -A[:, curind :(curind + len_s)]
        for blockSize in K['s']:
            blocknum += 1
            list_row = [str(x + 1) for x in
                        list(A_s[:, offset:(offset + blockSize * blockSize)].
                             nonzero()[0])]
            list_col = list(A_s[:, offset:(offset + blockSize * blockSize)].
                            nonzero()[1])
            list_val = [x for x in
                        list(A_s[:, offset:(offset + blockSize * blockSize)].
                             data)]
            length = len(list_row)
            for i in range(length):
                setCol_row = (list_col[i] // blockSize) + 1
                setCol_col = (list_col[i] % blockSize) + 1
                if setCol_row <= setCol_col:
                    fp.write(" ".join((list_row[i], str(blocknum),
                                       str(setCol_row), str(setCol_col),
                                       str(list_val[i]))) + "\n")
            offset += blockSize * blockSize

    fp.close()
    return     
    

