#!/bin/sh

#  train_HOMF.sh
#

for ptype in 'exp' 'linear'
 do
    for ((window=2;window<=8;window=window+2))
     do
        for lam in 0.0001 0.001 0.01 0.1 1.0
        do
        echo $lam,$window,$ptype
        python HOMF.py -ptype $ptype -k 10 -maxit 20 -T $window -cg 10 -l $lam -train Train.csv -val Val.csv -thr 5
        done
     done
done
