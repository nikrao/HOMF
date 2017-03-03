#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:29:18 2017

@author: raon
"""

'''
script to create timing plots
'''

# cyclic ALS for MF
import numpy as np
import numpy.random as nr
import utilHOMF as uh
import time
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.sparse as ss

k = 10                  # RANK
lam = 0.1               # REGULARIZER
cgiter = 10            # ITERATIONS OF CONJUGATE GRADIENT
max_iter = 1           # ITERATIONS OF COORDINATE DESCENT (EPOCHS)
srow,scol = None,None   # LOCATION OF ROW AND COLUMN GRAPHS
alpha = 1               # TRADEOFF BETWEEN GRAPH AND RATINGS
ptype = 'linear'        # TRANSITION PROBABILITY FUNCTION
thresh = 5              # THRESHOLD TO DETERMINE SUCCESS
T = 4                   # WALK LENGTH TO USE
numtests = 10           # number of tests to average over
train = '/home/ubuntu/ML1m/Train.csv'
#mfrac = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1] # FRACTION OF TRAINING DATA TO USE
pfrac = [1,2,5,10,15,20,25,30,35,40,45,50]
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
    
    
#%% MAIN ALGORITHM
print('loading data ...')
print('train')
Rtr = uh.load_data(train)
numuser = Rtr.shape[0]
print('transforming: %s'%(ptype))
Rtr = uh.function_transform(Rtr,ptype=ptype)
final_time = []
for proc in pfrac:
    print('starting HOMF with k {} T {} lam {} proc {}'.format(k,T,lam,proc))    
    #nrow,ncol= Rtr.shape
    #keepr = range(int(np.ceil(m*nrow)))
    #keepc = range(int(np.ceil(m*ncol)))
    #Rtr = ss.csr_matrix(Rtr)
    #Rtmp = Rtr[keepr,:]
    #Rtmp = Rtmp[:,keepc]
    A=uh.make_A_si(Rtr)
    p = A.shape[0]
    U,V = initvars(p,k,np.sqrt(k))
    bu,bv = np.zeros((p,)),np.zeros((p,))
    idset = range(p)
    init_time = time.time()
    for test in range(numtests):
        print('test {}'.format(test))
        P = Pool(proc)
        for t in range(max_iter):
            Vlist = P.map(update_allcols,idset,[U for i in range(p)])
            for i in range(len(Vlist)):
                V[Vlist[i][2],:] = Vlist[i][0]
                bv[Vlist[i][2]]  = Vlist[i][1]    
            Ulist = P.map(update_allrows,idset,[V for i in range(p)])
            for i in range(len(Ulist)):
                U[Ulist[i][2],:] = Ulist[i][0]
                bu[Ulist[i][2]]  = Ulist[i][1]
    final_time.append(time.time() - init_time)
    print(final_time)
    
print(final_time)
P.close()