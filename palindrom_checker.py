import numpy as np
import csv 
from sklearn.model_selection import KFold

def initialize_parameters(num_of_neurons):
    np.random.seed(23)
    
W1=np.random.uniform(0, 32, (10,2))
W2=np.random.uniform(0, 32, (2,1))
H1=np.random.uniform(0, 32, (1,2))
# X= some csv file

    
def sig(UW):
    UW=1/(1+np.exp(UW))
    return UW

def forward(inp):
    X = [[], []]
    X[0]=np.pad(inp[0:5], (0, 5), 'constant', constant_values=(0, 0))
    X[1]=np.pad(inp[5:10], (5, 0), 'constant', constant_values=(0, 0))
    H1[0]=np.dot(X[0],W1[0])
    H1[1]=np.dot(X[1],W1[1])
    o1= np.dot(H1,W2)
    UW=sig(o1)
    return UW
    
def backward(y_hat,instance):
    learning_rate=0.5
    W2=W2-learning_rate*(y_hat-instance[10])*H1
    
    W1=W1-0.5*np.concatenate((instance[0:10], instance[0:10]), axis = 1)


  
def main():
    # initialization place
    with open('palindrom_data.csv',mode='r') as file:
        csvFile=csv.reader(file)
        kf = KFold(n_splits = 4)
        for train_index, test_index in kf.split(csvFile):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = csvFile[train_index], csvFile[test_index]
        for index, row in X_train.iterrows():
            instance = row[:-1]
            target = row[-1]
            o1=forward(instance)
            Loss = -instance[10]*np.log(o1)-(1-instance[10])*np.log(1-o1)
            backward(o1,instance)