#!/bin/sh

#  train_HOMF.sh
#
# /home/ubuntu/HOMF/Data/ML20m
# /Users/raon/Desktop/Projects_2016/GraphEmbeddings/node-embeddings/Data/ML1m

for lam in 0.001 0.01 0.1 1 10
do
echo $lam
    python MF.py -k 10 -maxit 20 -cg 10 -l $lam -train /home-local/raon/Data/FilmTrust/Train.csv -val /home-local/raon/Data/FilmTrust/Train.csv
done

