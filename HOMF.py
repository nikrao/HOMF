#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:42:25 2016

@author: raon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:55:41 2016

@author: raon
"""

# cyclic ALS for MF
import numpy as np
import numpy.random as nr
#import utilHOMF100k as uh
import utilHOMF as uh

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
        
        
#%% main stuff
if __name__=='__main__':
    k = 10                  # RANK
    lam = 0.1               # REGULARIZER
    T = 4                   # LENGTH OF WALK
    cgiter = 100            # ITERATIONS OF CONJUGATE GRADIENT
    max_iter = 10           # ITERATIONS OF COORDINATE DESCENT (EPOCHS)
    srow,scol = None,None   # LOCATION OF ROW AND COLUMN GRAPHS
    alpha = 1               # TRADEOFF BETWEEN GRAPH AND RATINGS
    ptype = 'linear'        # TRANSITION PROBABILITY FUNCTION
    thresh = 5              # THRESHOLD TO DETERMINE SUCCESS
    nproc  = 16             # NUMBER OF PROCESSORS TO USE
    
    import sys
    foo= sys.argv
    for i in range(1,len(foo)):
        if foo[i]=='-k':    k = int(float(foo[i+1]))        
        if foo[i]=='-train':train = foo[i+1]
        if foo[i]=='-val':  val = foo[i+1]
        if foo[i]=='-siderow':srow = foo[i+1]
        if foo[i]=='-sidecol':scol = foo[i+1]
        if foo[i]=='-maxit': max_iter = int(float(foo[i+1]))            
        if foo[i]=='-T':    T = int(float(foo[i+1]))
        if foo[i]=='-cg':   cgiter = int(float(foo[i+1]))
        if foo[i]=='-l':    lam = float(foo[i+1])
        if foo[i]=='-ptype':ptype = foo[i+1]
        if foo[i]=='-alpha':alpha = float(foo[i+1])
        if foo[i]=='-thr':thresh = float(foo[i+1])
        if foo[i]=='-nproc':nproc = int(float(foo[i+1]))
        
    savefile = ptype+'_'+str(lam)+'_'+str(T)+'_'+str(k)+'_'+str(alpha)+'.csv'
   
    ########
    print('loading data ...')
    print('train')
    Rtr = uh.load_data(train)
    numuser = Rtr.shape[0]
    print('transforming: %s'%(ptype))
    Rtr = uh.function_transform(Rtr,ptype=ptype)
    print('creating transition prob matrix')
    if srow!=None or scol!=None:
        A=uh.make_A_si(Rtr,alpha=alpha,rowlink=srow,collink=scol)
    else:
        A=uh.make_A_si(Rtr)
    
    p = A.shape[0]
    print('A has {} rows'.format(p))
    print('validation')
    Rv  = uh.load_data(val)
    print('Initializing')
    U,V = initvars(p,k,np.sqrt(k))
    bu,bv = np.zeros((p,)),np.zeros((p,))
    print('starting HOMF with k {} T {} lam {}'.format(k,T,lam))
    print('cyclic CD for %d iterations'%(max_iter))

    p5,p10 = [],[];    
    
    from pathos.multiprocessing import ProcessingPool as Pool
    idset = range(p)
    
    P = Pool(nproc)
    preds = {}
    print('Initial Values')
    for t in range(max_iter):
        from functools import partial
        print('Iter %d'%(t+1))       
        func = partial(update_allcols,U=U)
        Vlist = P.map(func,idset,chunksize=p/nproc)
        #Vlist = P.map(update_allcols,idset,[U for i in range(p)],chunksize=p/nproc)
        for i in range(len(Vlist)):
            V[Vlist[i][2],:] = Vlist[i][0]
            bv[Vlist[i][2]]  = Vlist[i][1]
        func = partial(update_allrows,V=V)
        Ulist = P.map(func,idset,chunksize=p/nproc)
        #Ulist = P.map(update_allrows,idset,[V for i in range(p)],chunksize=p/nproc)
        for i in range(len(Ulist)):
            U[Ulist[i][2],:] = Ulist[i][0]
            bu[Ulist[i][2]]  = Ulist[i][1]
        
        tmp = uh.predict(U,bu,Rv,numuser)
        p5.append(uh.Calculate(tmp,n=5,thr=thresh))
        p10.append(uh.Calculate(tmp,n=10,thr=thresh))
       
    P.close()
    
    f = open(str(5)+savefile,'w')
    f.write('\n'.join('%s %s %s %s %s' % x for x in p5))
    f.close()
    f = open(str(10)+savefile,'w')
    f.write('\n'.join('%s %s %s %s %s' % x for x in p10))
    f.close()
    
    print('done')
