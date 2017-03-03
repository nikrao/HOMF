#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:50:10 2017

@author: raon
"""

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
        keep = nr.choice(inds,int(np.ceil(frac*len(inds))))
        Rout[i,keep] = R[i,keep]
    Rout = ss.coo_matrix(Rout)
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
        
    
def update_allcolsMF(ids,U):
    a = uh.colsamplemf(Rtr,ids)
    v,biasv = uh.colupdate(a,U,lam,cgiter)
    return (v,biasv,ids)
    
def update_allrowsMF(ids,V):
    a = uh.rowsamplemf(Rtr,ids)
    u,biasu = uh.rowupdate(a,V,lam,cgiter)
    return (u,biasu,ids)
    
    
#%% parameters    
k = 10    
fracset= [.1,.2,.3,.4,.5,.6,.7,.8]
Tset = [2,3,4,5]
lam = 0.01
cgiter = 10
max_iter = 30
ptype = 'exp'
savefile ='results.txt'
train = '/home-local/raon/Data/ML1m/Train.csv'
test  = '/home-local/raon/Data/ML1m/Test.csv'
#train = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/ML1m/Train.csv'
#test  = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/ML1m/Train.csv'
BESTVALS = []
BESTMF = []

# %% main algorithm
if __name__=='__main__':
    Rv  = uh.load_data(test)
    for T in Tset:    
        for f in fracset:
            print('HOMF T = {}, sampling fraction = {}'.format(T,f))
            
            Rtr = uh.load_data(train)
            Rtr = subsample(Rtr,f)
            numuser = Rtr.shape[0]
            Rtr = uh.function_transform(Rtr,ptype=ptype)
            A=uh.make_A_si(Rtr)
            At = uh.make_A_si(Rv)
            p = A.shape[0]
            U,V = initvars(p,k,np.sqrt(k))
            bu,bv = np.zeros((p,)),np.zeros((p,))
        
            p5 = []        
    
            idset = range(p)
            
            P = Pool()
            print('Initial Values')
            for t in range(max_iter):
                print('Iter %d'%(t+1))        
                Vlist = P.map(update_allcols,idset,[U for i in range(p)])
                for i in range(len(Vlist)):
                    V[Vlist[i][2],:] = Vlist[i][0]
                    bv[Vlist[i][2]]  = Vlist[i][1]    
                Ulist = P.map(update_allrows,idset,[V for i in range(p)])
                for i in range(len(Ulist)):
                    U[Ulist[i][2],:] = Ulist[i][0]
                    bu[Ulist[i][2]]  = Ulist[i][1]
                
                p5.append(uh.TestRMSE(U,V,At,T))
                
            
            # best val            
            BESTVALS.append(min(p5)) 
    
            
                
            ''' do MF only for one T '''
            if T==Tset[0]:
                print('MF , sampling fraction = {}'.format(f))
                m,n = Rtr.shape
                U = nr.randn(m,k)/np.sqrt(k)
                V = nr.randn(n,k)/np.sqrt(k)
                bu,bv = np.zeros((m,)),np.zeros((n,))
                p5 = []
                idsetu = range(m)
                idsetv = range(n)
                for t in range(max_iter):
                    print('Iter %d'%(t+1))        
                    Vlist = P.map(update_allcolsMF,idsetv,[U for i in range(n)])
                    for i in range(len(Vlist)):
                        V[Vlist[i][2],:] = Vlist[i][0]
                        bv[Vlist[i][2]]  = Vlist[i][1]    
                    Ulist = P.map(update_allrowsMF,idsetu,[V for i in range(m)])
                    for i in range(len(Ulist)):
                        U[Ulist[i][2],:] = Ulist[i][0]
                        bu[Ulist[i][2]]  = Ulist[i][1]
                    
                    p5.append(uh.TestRMSE_MF(U,V,Rv))
                BESTMF.append(min(p5))
            
            with open(savefile,'w') as sf:
                sf.write('\n'.join('%s' % x for x in BESTVALS))
                sf.write('\n MATRIX FACTORIZATION \n')
                sf.write('\n'.join('%s' % x for x in BESTMF))
            
            print(BESTVALS)
            print(BESTMF)        
    
            
            
            
    print(BESTVALS)
    print(BESTMF)        
            
            
    P.close()

