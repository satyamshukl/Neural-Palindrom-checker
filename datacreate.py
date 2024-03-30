import numpy as np
import itertools as it
import csv

all_sequence=[seq for seq in it.product("01", repeat=10)]
with open('train.csv', mode="w") as file:
    with open('test.csv', mode="w") as file1:
        for index,row in enumerate(all_sequence):
            if(index<512):
                csvwriter=csv.writer(file)
                csvwriter.writerow(row)
            else:
                csvwriter=csv.writer(file1)
                csvwriter.writerow(row)
            
            