import numpy as np
import pandas as pd
from os import listdir

def align_cdr3(cdr3):
    l = len(cdr3)
    seq = ['.']*max_cdr3_length
    for i in range(int(np.floor(l/2))):
        if i == 0:
            continue
        seq[i] = cdr3[i]
        seq[max_cdr3_length - 1 - i] = cdr3[l - 1 - i]
    if l%2 != 0:
        seq[int(np.floor(l/2))] = cdr3[int(np.floor(l/2))]
    return ''.join(seq)

def encode(seq):
    res = np.zeros(len(seq)*21)
    for i, c in enumerate(seq):
        res[21*i + mapping_dic[c]] = 1
    return res

def encode_row(row):
    v_aa = ''
    v_aa = v_dic[row["vb"]]
    cdr3 = encode(align_cdr3(row["cdr3b"]))
    cdr1 = encode(v_aa[cdr1_bounds[0]:cdr1_bounds[1]])
    cdr2 = encode(v_aa[cdr2_bounds[0]:cdr2_bounds[1]])
    return(np.concatenate((cdr1, cdr2, cdr3)))

def read_v_aa():
    v_dic = {}
    f = open('v_aa.txt')
    line = f.readline().strip("\n")
    v_gene = line.split("|")[1]
    aas = ''
    counter = 1
    while(line != ''):
        if (counter-1)%3 == 0 and counter != 1:
            aas = ''
            line = f.readline().strip("\n")
            if line == '':
                break
            v_gene = line.split("|")[1]
        elif (counter-1)%3 != 0:
            aas = aas + f.readline().strip("\n")
            if counter%3 == 0:
                v_dic[v_gene] = aas
        counter += 1
    f.close()
    return v_dic
    

path = '/home/tmazumd1/TCRGP-master/training_data/paper/'
fs = listdir(path)
fs = [f for f in fs if 'unique' in f and 'human' in f and "vdj" in f]
max_cdr1_length = 12
max_cdr2_length = 10
max_cdr3_length = 24 - 2
max_length = sum([max_cdr1_length, max_cdr2_length, max_cdr3_length])
cdr1_bounds = [27-1, 38] #zero indexing and inclusive of upper bound
cdr2_bounds = [56-1, 65]
mapping_dic = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,
               'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,
               'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,
               'W':18,'Y':19,'.':20}
v_dic = read_v_aa()

count = 0
for i, f in enumerate(fs):
    data = pd.read_csv(path + f, dtype = str)
    data = data.loc[data["epitope"] != "none"]
    if len(data) < 100:
        continue
    data = data.iloc[0:100,:]
    data['encoding'] = data.apply(encode_row, axis=1)
    x = np.zeros((100, max_length*21))
    for j in range(len(data)):
        x[j,:] = data.loc[j,'encoding']    
    y = np.ones(100)*count
    if count == 0:
        X = x
        Y = y
    if count != 0:
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))
    count += 1

print(count)
np.save("dataset/vdjdb/X", X)
np.save("dataset/vdjdb/Y", Y)

