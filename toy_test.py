#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:22:12 2017

@author: raon
"""

'''
toy data generation and testing
'''

import numpy as np
import numpy.random as nr
import utilHOMF as uh
import scipy.sparse as ss
from pathos.multiprocessing import ProcessingPool as Pool

def makemat(m,k):
    U = nr.rand(m,k)
    V = nr.rand(m,k)
    R = U.dot(V.T)
    return R
    
def subsample(R,frac):
    m,n = R.shape
    R = ss.csr_matrix(R)
    Rout = ss.lil_matrix(R.shape)
    for i in range(m):
        row = R[i,:]
        inds = row.nonzero()
        inds = inds[1]
        keep = nr.choice(inds,np.ceil(frac*len(inds)))
        Rout[i,keep] = R[i,keep]
    return Rout

    
#%% initialize variables
def initvars(p,k,rho=0.01):
    U = nr.randn(p,k)/rho
    V = nr.randn(p,k)/rho
    return U,V
  

#%% all udpate functions to invoke later
def update_allcols(ids,U):
    a = uh.colsample(A,ids,T)
    v,biasv = uh.colupdate(a,U,lam,cgiter)
    return (v,biasv,ids)
    
def update_allrows(ids,V):
    a = uh.rowsample(A,ids,T)
    u,biasu = uh.rowupdate(a,V,lam,cgiter)
    return (u,biasu,ids)
        
#%% parameters
    
m = 500 
k = 10    
fracset = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
ntests = 3
Tset = [1,2,3,4,5]
lamset = [0.001,0.01,0.1,1]
cgiter = 10
maxiter = 10
ptype = 'exp'
savefile ='results.txt'

HOMFPRE = []
for T in Tset:
    for frac in fracset:
        homferr=np.zeros((len(lamset),ntests))
        for test in range(ntests):
            print('fraction {} test {}'.format(frac,test))
            # data
            R = makemat(m,k)
            # make train test
            Rtr,Rv = subsample(R,frac)
            Rv = ss.coo_matrix(Rv)
            numuser = Rtr.shape[0]
            Rtr = uh.function_transform(Rtr,ptype=ptype)
            A=uh.make_A_si(Rtr)
            p = A.shape[0] 
            for lind in range(len(lamset)):
                lam = lamset[lind]
                print('.')

                U,V = initvars(p,k,np.sqrt(k))
                bu,bv = np.zeros((p,)),np.zeros((p,))
                idset = range(p)
                P = Pool()
                tempval = 0
                for t in range(maxiter):
                    Vlist = P.map(update_allcols,idset,[U for i in range(p)])
                    for i in range(len(Vlist)):
                        V[Vlist[i][2],:] = Vlist[i][0]
                        bv[Vlist[i][2]]  = Vlist[i][1]    
                    Ulist = P.map(update_allrows,idset,[V for i in range(p)])
                    for i in range(len(Ulist)):
                        U[Ulist[i][2],:] = Ulist[i][0]
                        bu[Ulist[i][2]]  = Ulist[i][1]
                    
                    tmp = uh.predict(U,bu,Rv,numuser)
                    tmp = uh.Calculate(tmp,n=5,thr=4)
                    tempval = max(tempval,tmp[0])
                homferr[lind,test] = tempval
        HOMFPRE.append((T,frac,np.max(np.mean(homferr,axis=1))))
        
        print(HOMFPRE)
        np.savetxt(savefile,HOMFPRE) 
P.close()

#%% add comparison to vanilla MF
'''
def update_allcols(ids,U):
    a = uh.colsamplemf(A,ids)
    v,biasv = uh.colupdate(a,U,lam,cgiter)
    return (v,biasv,ids)
    
def update_allrows(ids,V):
    a = uh.rowsamplemf(A,ids)
    u,biasu = uh.rowupdate(a,V,lam,cgiter)
    return (u,biasu,ids)
    
MFPRE = []
for frac in fracset:
    mferr=np.zeros((len(lamset),ntests))
    for test in range(ntests):
        print('fraction {} test {}'.format(frac,test))
        # data
        R = makemat(m,k)
        # make train test
        Rtr,Rv = subsample(R,frac)
        Rv = ss.coo_matrix(Rv)
        numuser = Rtr.shape[0]
        for lind in range(len(lamset)):
            lam = lamset[lind]
            print('.')
            idsetu = range(m)
            idsetv = range(m)
            P = Pool()
            preds = {}
            print('Initial Values')
            for t in range(maxiter):
                print('Iter %d'%(t+1))        
                Vlist = P.map(update_allcols,idsetv,[U for i in range(m)])
                for i in range(len(Vlist)):
                    V[Vlist[i][2],:] = Vlist[i][0]
                    bv[Vlist[i][2]]  = Vlist[i][1]    
                Ulist = P.map(update_allrows,idsetu,[V for i in range(m)])
                for i in range(len(Ulist)):
                    U[Ulist[i][2],:] = Ulist[i][0]
                    bu[Ulist[i][2]]  = Ulist[i][1]
                
                tmp = uh.predictuv(U,V,bu,bv,Rv)
                tmp = uh.Calculate(tmp,n=2,thr=4)
                tempval = max(tempval,tmp[0])
            mferr[lind,test] = tempval
    MFPRE.append(np.max(homferr))
    print(HOMFPRE)
        
P.close()
    
'''       
        
        
        
        
        
        
        