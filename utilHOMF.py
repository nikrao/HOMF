#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:08:57 2016

@author: raon
"""
import scipy.sparse as ss
#THE FUNCTIONS BELOW SAMPLE A COLUMN OR ROW OF MATRIX

def colsamplemf(A,colinds):
    A = ss.csc_matrix(A)
    a = A[:,colinds]
    return a.toarray()


def rowsamplemf(A,rowinds):
    A = ss.csr_matrix(A)
    a = A[rowinds,:]
    return a.toarray()




#THE FUNCTIONS BELOW RETURN VECTOR OF THE FORM
#a + Aa + A^2a ... for columns and rows

def colsample(A,colinds,T):
    A = ss.csc_matrix(A)
    a1 = A[:,colinds]
    v = A[:,colinds]
    for t in range(T-1):
        v = a1 + A*v
    return v.toarray()


def rowsample(A,rowinds,T):
    A = ss.csr_matrix(A)
    a1 = A[rowinds,:]
    v = A[rowinds,:]
    for t in range(T-1):
        v = a1 + v*A
    return v.toarray()


#THE FUNCTIONS BELOW PERFORM RIDGE UPDATES 
#OF ONLY COLUMN OR ROW VARIABLES

from sklearn.linear_model import Ridge    
def colupdate(y,U,lam,cgiter=100):
    y = np.ravel(y)
    ids=np.ravel(np.argwhere(y!=0))
    if len(ids)>0:
        clf = Ridge(alpha=lam,max_iter=cgiter,solver = 'sparse_cg',fit_intercept=True)
        clf = clf.fit(U[ids,:],y[ids])
        vhat = clf.coef_
        bias = clf.intercept_
    else:
        bias = 0
        vhat = np.zeros((U.shape[1],))
    return vhat,bias
    
def rowupdate(y,V,lam,cgiter=100):
    y = np.ravel(y)
    ids=np.ravel(np.argwhere(y!=0))
    if len(ids)>0:
        clf = Ridge(alpha=lam,max_iter=cgiter,solver = 'sparse_cg',fit_intercept=True)
        clf = clf.fit(V[ids,:],y[ids])
        uhat = clf.coef_
        bias = clf.intercept_
    else:
        bias = 0
        uhat = np.zeros((V.shape[1],))
    return uhat,bias
    

    
#THE FOLLOWING FUNCTIONS PREPS THE DATA INTO SPARSE MATRICES
def load_data(fname):
    c = 0
    with open(fname) as f:
        row,col,data = [],[],[]
        for line in f:
            if c==0:
                vals = line.strip('\n').split(',')
                num_rows = int(vals[0])
                num_cols = int(vals[1])
                c+=1                
            else:
                vals = line.strip('\n').split(',')
                rowval = int(float(vals[0]))
                colval = int(float(vals[1]))
                row.append(rowval)
                col.append(colval)
                data.append(float(vals[2]))
        
    X = ss.coo_matrix((data,(row,col)),shape=(num_rows,num_cols))
    return X
                        
def make_A_nosi(X):
    from sklearn.preprocessing import normalize
    X = ss.csr_matrix(X)
    X1 = normalize(X,norm='l1',axis=1)
    X = ss.csc_matrix(X)
    X2 = normalize(X,norm='l1',axis=0)
    A = ss.bmat([[None, X1],[X2.T,None]])
    return A
    
def make_A_sym(X):
    from sklearn.preprocessing import normalize
    X = ss.csr_matrix(X)
    X1 = normalize(X,norm='l1',axis=1)
    return X1
    
def make_A_si(X,alpha=1,rowlink=None,collink=None):
    if rowlink==None and collink==None:
        A = make_A_nosi(X)
        return A
    RL,RC = None,None
    if rowlink!=None:
        c=0
        with open(rowlink) as f:
            row,col,data = [],[],[]
            for line in f:
                if c==0:
                    vals = line.strip('\n').split(',')
                    p = int(vals[0])
                    c+=1
                else:
                    vals = line.strip('\n').split(',')
                    rowval = int(float(vals[0]))
                    colval = int(float(vals[1]))
                    row.append(rowval)
                    col.append(colval)
                    data.append(float(vals[2]))
                    row.append(colval)
                    col.append(rowval)
                    data.append(float(vals[2]))
        RL = ss.coo_matrix((data,(row,col)),shape=(p,p))
        RL = RL*(1-alpha)
    if collink!=None:
        c=0
        with open(collink) as f:
            row,col,data = [],[],[]
            for line in f:
                if c==0:
                    vals = line.strip('\n').split(',')
                    p = int(vals[0])
                    c+=1
                else:
                    vals = line.strip('\n').split(',')
                    rowval = int(float(vals[0]))
                    colval = int(float(vals[1]))
                    row.append(rowval)
                    col.append(colval)
                    data.append(float(vals[2]))
                    row.append(colval)
                    col.append(rowval)
                    data.append(float(vals[2]))
        RC = ss.coo_matrix((data,(row,col)),shape=(p,p))
        RC = RC*(1-alpha)
    A = ss.bmat([[RL, X*alpha],[X.T*alpha,RC]])
    from sklearn.preprocessing import normalize
    A = normalize(A,norm='l1',axis=1)
    return A
                
                
    

#THE NEXT FUNCTIONS RETURNS PREDICTIONS
def predict(U,bu,Test,nrows):
    data = Test.data
    rows = Test.row
    cols = Test.col
    TUP = []
    for c in range(len(data)):
        s = sum(U[rows[c],:]*U[cols[c]+nrows,:]) + bu[rows[c]] + bu[cols[c] + nrows]
        TUP.append((rows[c],cols[c],data[c],s))
    return TUP
    
def predictuv(U,V,bu,bv,Test):
    data = Test.data
    rows = Test.row
    cols = Test.col
    TUP = []
    for c in range(len(data)):
        s = sum(U[rows[c],:]*V[cols[c],:]) + bu[rows[c]] + bv[cols[c]]
        TUP.append((rows[c],cols[c],data[c],s))
    return TUP
    
#THE FUNCTIONS BELOW CREATE THE "f(X)" matrices
def function_transform(R,ptype='linear'):
    if ptype=='linear':
        return R
    elif ptype=='exp':
        d = R.data
        d = np.exp(d)
        R.data = d
        return R
    elif ptype=='step':
        d = np.ones(R.data().shape)
        R.data = d
        return R

# THE FUNCTIONS BELOW CALCULATE THE METRICS WE CARE ABOUT
#######################################################################################################################
# cTest has to be a dataframe, U is n*10 matrix while V is a 10*n matrix
# cTest which is stored as a dataframe is converted into an array and later dictionary with the topN values calculated
# using U and V 
# For caclulating precision, recall, MAP and second NDCG measurement we threshold the relevant items as those which have a rating of 5
# Precision is basically the average of total number of relevant recommendations by the top n recommendations for each user
# Recall is the number of relevant items in the top n recommendations divided by the total number of relevant items (which can be maximum of n)
# Average Precision is the average of precision at which relevant items are recorded among the top n recommendations. 
# MAP is the mean of the average precision over all the users
# NDCG is normal discounted cumulative gain. IDCG is calculated based on the actual top N recommendations while DCG is calculated 
# based on the predicted top N. NDCG = DCG/IDCG
# NDCG@N applies to 2**x - 1 function on each rating before multiplying top ith item by 1/log2(i+1)
######################################################################################################################
import numpy as np
from collections import defaultdict 
#from TopN import Calculate


def parsefile(filename):
    """
    assuming that we are reading results from saved prediction score file
    each line:
    userId, movieId, actrualstars, predicted_score
    """
    dic = {}
    f = open(filename)
    for l in f:
        c = l.strip().split(',')
        uid = c[0]
        mid = c[1]
        entry = {}
        entry['t'] = float(c[2]) # true score
        entry['p'] = float(c[3]) # pred score
        if uid not in dic:
            dic[uid] = {}
        dic[uid][mid] = entry
    f.close()
    return dic

def cal_precision(dicTopn, n, thr):
    """
    re-writing the precision calculation
    """
    def getkey(tp):
        return tp[1]
    num_good_user = 0.0
    Prec =0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        x_sorted = sorted(x, key=getkey, reverse = True)        
        sumP = 0.0
        num_good_user +=1.0
        for i in range(n):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
        Prec += sumP/n
    if num_good_user<1.0:
        print('no valid users, ERROR metric')
        return 0.0
    Prec = Prec/num_good_user
    return Prec    

def cal_recall(dicTopn, n, thr):
    """
    re-writing the recall caculation
    """
    def getkey(tp):
        return tp[1]
    num_good_user = 0.0
    Rec =0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        act_tot = 0.0
        for i in range(len(x)):
            if x[i][0]>=thr:
                act_tot += 1.0
        if act_tot < 1.0:
            continue # skip users without '1''s in ground truth
        x_sorted = sorted(x, key=getkey, reverse = True)
        sumP = 0.0
        num_good_user +=1.0
        for i in range(n):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
        #Rec += float(sumP)/min(n, act_tot)
        # below change acc to vs
        Rec += float(sumP)/act_tot
    if num_good_user<1.0:
        print('no valid users, ERROR metric')
        return 0.0
    Rec = Rec/num_good_user
    return Rec
    
def cal_map(dicTopn, n, thr):
    """
    re-writing the MAP function
    """
    def getkey(tp):
        return tp[1]
    MAP = 0.0
    num_good_user = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        act_tot = 0.0
        for i in range(len(x)):
            if x[i][0]>=thr:
                act_tot += 1.0
        if act_tot < 1.0:
            continue # skip users without '1''s in ground truth
        x_sorted = sorted(x, key=getkey, reverse = True)
        sumP = 0.0
        ap = 0.0
        num_good_user +=1.0
        upper = min(n, len(x))
        for i in range(upper):
            if x_sorted[i][0] >= thr:
                sumP += 1.0
                ap += sumP/float(i+1.0)
        MAP += ap/min(upper, act_tot)
    if num_good_user<1.0:
        print('no valid users, ERROR metric')
        return 0.0
    MAP = MAP/num_good_user
    return MAP

def cal_ndcg_type1(dicTopn, n, thr):
    def getkeydcg(tp):
        return tp[1]  # predicted
    def getkeyidcg(tp):
        return tp[0]  # true
    NDCG = 0.0
    num_good_user = 0.0
    for uid in dicTopn:
        z = dicTopn[uid]
        if len(z) < n:
            continue  # skip users with less than n ratings
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        dcg = 0.0
        idcg = 0.0
        num_good_user += 1.0
        sorted_x1 = sorted(x, key=getkeydcg, reverse = True)
        for i in range(n):
            dcg +=(2**sorted_x1[i][0]-1)/np.log2(i+2.0)
        sorted_x2 = sorted(x, key=getkeyidcg, reverse = True)
        for i in range(n):
            idcg += (2**sorted_x2[i][0] -1)/np.log2(i+2.0)
        NDCG  += dcg/idcg
    if num_good_user<1.0:
        print('no valid users, ERROR metric')
        return 0.0
    NDCG = NDCG/num_good_user
    return NDCG
    
def cal_ndcg_type2(dicTopn,  thr):
    def getkeydcg(tp):
        return tp[1]  # predicted
    def getkeyidcg(tp):
        return tp[0]  # true
    NDCG = 0.0
    num_good_user = 0.0
    for uid in dicTopn:
        z = dicTopn[uid] 
        x = [(z[mid]['t'], z[mid]['p']) for mid in z]
        dcg = 0.0
        idcg = 0.0
        num_good_user += 1.0
        n = len(x)
        sorted_x1 = sorted(x, key=getkeydcg, reverse = True)
        for i in range(n):
            if sorted_x1[i][0] >= thr:
                dcg += 1./np.log2(i+2.0)
        sorted_x2 = sorted(x, key=getkeyidcg, reverse = True)
        for i in range(n):
            if sorted_x2[i][0]>=thr:
                idcg += 1./np.log2(i+2)
        if idcg ==0.0:
            NDCG += 0.0
        else:
            NDCG += dcg/idcg
    if num_good_user<1.0:
        print('no valid users, ERROR metric')
        return 0.0
    NDCG = NDCG/num_good_user
    return NDCG
    
def Calculate_fromFile(filename, n=10, thr = 5):
    """
    assuming that we are reading results from saved prediction score file
    each line:
    userId, movieId, actrualstars, predicted_score
    """
    dicTopn = parsefile(filename)
    OutPrec = cal_precision(dicTopn,n,thr)
    OutRec = cal_recall(dicTopn,n, thr)
    OutMAP = cal_map(dicTopn, n, thr)
    OutNDCG = cal_ndcg_type1(dicTopn, n, thr)
    OutNDCG2 = cal_ndcg_type2(dicTopn, thr)
    return OutPrec, OutRec, OutMAP, OutNDCG, OutNDCG2

def parseTUPS(tup):
    """
    assuming that we are reading results from saved prediction score file
    each line:
    userId, movieId, actrualstars, predicted_score
    """
    dic = {}
    for c in tup:
        uid = c[0]
        mid = c[1]
        entry = {}
        entry['t'] = float(c[2]) # true score
        entry['p'] = float(c[3]) # pred score
        if uid not in dic:
            dic[uid] = {}
        dic[uid][mid] = entry
    return dic
    
def Calculate(tup,n=10,thr=5):
    dicTopn = parseTUPS(tup)
    OutPrec = cal_precision(dicTopn,n,thr)
    OutRec = cal_recall(dicTopn,n, thr)
    OutMAP = cal_map(dicTopn, n, thr)
    OutNDCG = cal_ndcg_type1(dicTopn, n, thr)
    OutNDCG2 = cal_ndcg_type2(dicTopn, thr)
    return (OutPrec, OutRec, OutMAP, OutNDCG, OutNDCG2)

def TestRMSE(U,V,At,A,T):
    '''
    U,V = predicted latent factors. A = graph for test data
    '''
    n = V.shape[0]
    MSE = 0
    N = 0
    for i in range(n):
        a = colsample(At,i,T)
        atrue = colsample(A,i,T)
        ind = np.argwhere(a!=0)
        ind = list(ind[:,0])
        indtrue = np.argwhere(atrue!=0)
        indtrue = list(indtrue[:,0])
        to_keep  = [i for i in ind if i not in indtrue]
        N  = N + len(to_keep)
        Y    = a[to_keep]
        Yhat = np.dot(U[to_keep,:],V[i,:].T)
        MSE  += np.sum((Y-Yhat)**2)
    RMSE = np.sqrt(MSE/N)
    return RMSE
        
    
def TestRMSE_MF(U,V,A):
    '''
    A = test set (sparse). Currently this might not be scalable, but will work for ML1m
    '''

    Yhat = np.dot(U,V.T)
    rows = A.row
    cols = A.col
    data = A.data
    N    = len(rows)
    MSE  = 0
    for i in range(N):
        yhat = Yhat[rows[i],cols[i]]
        MSE += (data[i]-yhat)**2
    return np.sqrt(MSE/N)
    