#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:46:41 2016

@author: raon
"""

print('running best param values for all metrics')

import HOMF as h
import numpy as np
import utilHOMF as uh
from pathos.multiprocessing import ProcessingPool as Pool

fhead = '/home-local/raon/Results/'
fhead = '/home/ubuntu/'
#fhead = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Results/'
fdata = 'EpinSmall'
ftail = '/bv.csv'
fpath = fhead + fdata + ftail


def getallparam(fpath):
    D = {}
    with open(fpath) as f:
        for line in f:
            temp = line.strip('\n').split(':')
            key = eval(temp[0])
            value = eval(temp[1])
            D[key] = value
    return D.items()
            
'''
params =  ((10, 'linear', 2, 1), [(1e-05, [0, 1]), (0.001, [2, 3, 4])])    
precision @ K    type    T  alpha  lambda  metric   lambda   metric
'''
def getsubsetparam(params):
    left = params[0]  # this will always be one tuple
    right = params[1] # this can have multiple values
    plist = []
    for r in right:
        tmp = [i for i in left]
        tmp = tmp + [i for i in r]
        plist.append(tmp)
    return plist
    
        
#order = [p,r,m,n1,n2 at 5, then all at 10]
dhead = '/home-local/raon/Data/'
dhead = '/home/ubuntu/'
#dhead = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/'
trfile = dhead + fdata + '/Train.csv'
ttfile = dhead + fdata + '/Test.csv'


# params =  [10, 'linear', 2, 1e-05, [0, 1]]
# output = list with requisite metrics per iteration (total 10)
def tt(trfile,ttfile,params):
    ## PARAMETERS
    n = params[0]  # precision @k
    ptype = params[1]  
    T = params[2]
    alpha = params[3]
    lam = params[4]
    metrics = params[5]
    cgiter = 100
    k = 10
    maxit = 30    
    #srow = '/home-local/raon/Data/EpinSmall/User_edge.csv'
    # uncomment above for epinion
    srow = None # comment this for ML1m
    scol = None

    #### INITIALIZATIONS
    
    print('train')
    Rtr = uh.load_data(trfile)
    numuser = Rtr.shape[0]
    print('transforming: %s'%(ptype))
    Rtr = uh.function_transform(Rtr,ptype=ptype)
    print('creating transition prob matrix')
    if srow!=None or scol!=None:
        A=uh.make_A_si(Rtr,alpha=alpha,rowlink=srow,collink=scol)
    else:
        A=uh.make_A_si(Rtr)
    p = A.shape[0]
    print('validation')
    Rv  = uh.load_data(ttfile)
    print('Initializing')
    U,V = h.initvars(p,k,np.sqrt(k))
    bu,bv = np.zeros((p,)),np.zeros((p,))
    print('starting HOMF with k {} T {} lam {}'.format(k,T,lam))
    print('cyclic CD for %d iterations'%(maxit))

    # udpate functions to invoke later
    def update_allcols(ids,U):
       a = uh.colsample(A,ids,T)
       v,biasv = uh.colupdate(a,U,lam,cgiter)
       return (v,biasv,ids)
    
    def update_allrows(ids,V):
       a = uh.rowsample(A,ids,T)
       u,biasu = uh.rowupdate(a,V,lam,cgiter)
       return (u,biasu,ids)
    
    
    idset = range(p)

    print('Initial Values')
    
    v=[]
    
    for t in range(maxit):
        print('Iter %d'%(t+1))        
        Vlist = P.map(update_allcols,idset,[U for i in range(p)])
        for i in range(len(Vlist)):
            V[Vlist[i][2],:] = Vlist[i][0]
            bv[Vlist[i][2]]  = Vlist[i][1]    
        Ulist = P.map(update_allrows,idset,[V for i in range(p)])
        for i in range(len(Ulist)):
            U[Ulist[i][2],:] = Ulist[i][0]
            bu[Ulist[i][2]]  = Ulist[i][1]
        
        tmp = uh.predict(U,bu,Rv,numuser)
        tv = uh.Calculate(tmp,n=n,thr=3)
        toappend = [tv[i] for i in metrics]
        v.append(toappend)
        
    v = np.array(v)
    v = np.amax(v,axis=0)
    return [params,v]

def fullsweep():
    allparams = getallparam(fpath)
    sparams = []
    for entry in allparams:
        sparams.append(getsubsetparam(entry))
    
    print('starting full runs')
    
    values_list = []
    
    for outparams in sparams:
        for params in outparams:
            if fdata=='ML1m':
                params.insert(3,0)
            out = tt(trfile,ttfile,params)
            values_list.append(out)
            print(out)
    print(values_list)
    
    

if __name__=='__main__':
    P =Pool()
    fullsweep()
    P.close()
    
    
