#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:17:02 2017

@author: raon
"""

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
varying the number of processors
'''

import numpy as np
import numpy.random as nr
import utilHOMF as uh
import time
from pathos.multiprocessing import ProcessingPool as Pool

    
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
k = 10    
procset= [1,2,5,10,15,20]
T = 4
lam = 0.1
cgiter = 10
max_iter = 2
ptype = 'exp'
savefile ='resultsProc.txt'
train = '/home-local/raon/Data/EpinSmall/Train.csv'
test  = '/home-local/raon/Data/EpinSmall/Test.csv'
TIMES = []


# %% main algorithm
if __name__=='__main__':
    Rv  = uh.load_data(test)
    for npr in procset:
        print('number of processors = {}'.format(npr))
        Rtr = uh.load_data(train)
        numuser = Rtr.shape[0]
        Rtr = uh.function_transform(Rtr,ptype=ptype)
        A=uh.make_A_si(Rtr)
        p = A.shape[0]
        U,V = initvars(p,k,np.sqrt(k))
        bu,bv = np.zeros((p,)),np.zeros((p,))
        p5 = []
        idset = range(p)
        P = Pool(npr)
        init_time = time.time()
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
        
        TIMES.append(time.time() - init_time)            
        print(TIMES)            
    P.close()

