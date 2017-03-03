#!/bin/sh

#  train_HOMF.sh
#
# /home/ubuntu/HOMF/Data/ML20m
# /Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/ML1m
for ptype in 'exp'
 do
    for ((window=2;window<=8;window=window+2))
     do
        for lam in 0.0001 0.001 0.01 0.1 1.0
        do
        echo $lam,$window,$ptype
        python HOMF.py -ptype $ptype -k 10 -maxit 20 -T $window -cg 10 -l $lam -train /home/ubuntu/Jester/Train.csv -val /home/ubuntu/Jester/Val.csv -thr 7
        done
     done
done
