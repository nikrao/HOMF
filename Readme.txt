# HOMF
Higher Order Matrix Factorization with or without side information

USAGE:
python HOMF.py  -train <Train File> -val <Validation or Test File> 

TRAIN/TEST/GRAPH SIDE INFORMATION FILE FORMAT:
num_rows,num_cols,nnz
row_id1,col_id1,value1
row_id2,col_id2,value2



OPTIONAL INPUTS (with defaults):
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

The file utilHOMF has all the utility functions, including ones to load data, and update the columns


DEPENDENCIES:
pathos
numpy
scipy
scikit-learn

