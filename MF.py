#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:15:10 2016

@author: raon
"""

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
import utilHOMF100k as uh


#%% all udpate functions to invoke later
def update_allcols(ids,U):
    a = uh.colsamplemf(A,ids)
    v,biasv = uh.colupdate(a,U,lam,cgiter)
    return (v,biasv,ids)
    
def update_allrows(ids,V):
    a = uh.rowsamplemf(A,ids)
    u,biasu = uh.rowupdate(a,V,lam,cgiter)
    return (u,biasu,ids)
        
#%% main stuff
if __name__=='__main__':
    k = 10
    l = 0.1
    cgiter = 100
    max_iter = 10
    
    import sys
    foo= sys.argv
    for i in range(1,len(foo)):
        if foo[i]=='-k':    k = int(float(foo[i+1]))        
        if foo[i]=='-train':train = foo[i+1]
        if foo[i]=='-val':  val = foo[i+1]
        if foo[i]=='-maxit': max_iter = int(float(foo[i+1]))            
        if foo[i]=='-cg':   cgiter = int(float(foo[i+1]))
        if foo[i]=='-l':    lam = float(foo[i+1])
        
    savefile = 'MF'+'_'+str(lam)+'_'+str(k)+'.csv'
   
    ########
    print('loading data ...')
    print('train')
    A = uh.load_data(train)
    m,n = A.shape
    print('A has {} rows and {} columns'.format(m,n))
    print('validation')
    Rv  = uh.load_data(val)
    print('Initializing')
    U = nr.randn(m,k)/np.sqrt(k)
    V = nr.randn(n,k)/np.sqrt(k)
    bu,bv = np.zeros((m,)),np.zeros((n,))
    print('starting MF with k {} lam {}'.format(k,lam))
    print('cyclic CD for %d iterations'%(max_iter))

    p5,p10 = [],[];
    
    from pathos.multiprocessing import ProcessingPool as Pool
    idsetu = range(m)
    idsetv = range(n)
    P = Pool()
    preds = {}
    print('Initial Values')
    for t in range(max_iter):
        print('Iter %d'%(t+1))        
        Vlist = P.map(update_allcols,idsetv,[U for i in range(n)])
        for i in range(len(Vlist)):
            V[Vlist[i][2],:] = Vlist[i][0]
            bv[Vlist[i][2]]  = Vlist[i][1]    
        Ulist = P.map(update_allrows,idsetu,[V for i in range(m)])
        for i in range(len(Ulist)):
            U[Ulist[i][2],:] = Ulist[i][0]
            bu[Ulist[i][2]]  = Ulist[i][1]
        
        tmp = uh.predictuv(U,V,bu,bv,Rv)
        p5.append(uh.Calculate(tmp,n=5,thr=5))
        p10.append(uh.Calculate(tmp,n=10,thr=5))
       
    P.close()
    
        
    f = open('Test'+str(5)+savefile,'w')
    f.write('\n'.join('%s %s %s %s %s' % x for x in p5))
    f.close()
    f = open('Test'+str(10)+savefile,'w')
    f.write('results at 10 \n')
    f.write('\n'.join('%s %s %s %s %s' % x for x in p10))
    f.close()
    
    print('done')
    
