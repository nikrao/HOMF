#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:40:45 2016

@author: raon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:46:41 2016

@author: raon
"""

print('running best param values for all metrics')

import numpy as np
import numpy.random as nr
import utilHOMF as uh
from pathos.multiprocessing import ProcessingPool as Pool

fhead = '/home-local/raon/Results/'
fhead = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Results/'
fdata = 'FilmTrust'
ftail = '/bvMF.csv'
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
            

# params =  (10, [(100.0, [0, 1, 2, 3, 4])])    
def getsubsetparam(params):
    left = params[0]  # this will always be 5 or 10
    right = params[1] # this can have multiple values
    plist = []
    for r in right:
        tmp = [left] + [i for i in r]
        plist.append(tmp)
    return plist
    


        
#order = [p,r,m,n1,n2 at 5, then all at 10]
dhead = '/home-local/raon/Data/'
dhead = '/Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/'
trfile = dhead + fdata + '/Train.csv'
ttfile = dhead + fdata + '/Test.csv'


# params =  [10, 'linear', 2, 1e-05, [0, 1]]
# output = list with requisite metrics per iteration (total 10)
def tt(trfile,ttfile,params):
    N = params[0]
    lam = params[1]
    metrics = params[2]
    cgiter = 100
    k = 10
    maxit = 20    


    print('loading data ...')
    print('train')
    A = uh.load_data(trfile)
    m,n = A.shape
    print('A has {} rows and {} columns'.format(m,n))
    print('validation')
    Rv  = uh.load_data(ttfile)
    print('Initializing')
    U = nr.randn(m,k)/np.sqrt(k)
    V = nr.randn(n,k)/np.sqrt(k)
    bu,bv = np.zeros((m,)),np.zeros((n,))
    print('starting MF with k {} lam {}'.format(k,lam))
    print('cyclic CD for %d iterations'%(maxit))

    # udpate functions to invoke later
    def update_allcols(ids,U):
        a = uh.colsamplemf(A,ids)
        v,biasv = uh.colupdate(a,U,lam,cgiter)
        return (v,biasv,ids)
    
    def update_allrows(ids,V):
        a = uh.rowsamplemf(A,ids)
        u,biasu = uh.rowupdate(a,V,lam,cgiter)
        return (u,biasu,ids)
    
    
    idsetu = range(m)
    idsetv = range(n)

    print('Initial Values')
    
    v=[]
    
    for t in range(maxit):
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
        tv = uh.Calculate(tmp,n=N,thr=3)
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
            print(params)
            out = tt(trfile,ttfile,params)
            values_list.append(out)
            print(out)
        
    print(values_list)
    

if __name__=='__main__':
    P =Pool()
    fullsweep()
    P.close()
    
