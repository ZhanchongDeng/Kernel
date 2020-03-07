'''
This file contains all methods/class working with string kernels.
'''
import numpy as np
import pandas as pd

def loadData(fp):
    newfile = open(fp, 'r')
    newfile.seek(0)
    raw_strings = newfile.read().split("\n")[:-1]
    return np.array([np.array(entry.split(" "), dtype="str") for entry in raw_strings])

'''
Calculates common substring of length p in both s and t. If a string occurs a times in s
and b times in t, then increment total count by a * b.
@param p is the length of substring
@param s is the first string.
@param t is the econd string.
'''
def kernel(s, t, p):
    product = 0
    # Maps every substring to a number within s and t of length p
    map_s = generate_map(s, p)
    map_t = generate_map(t, p)
    # Iterate through one of them
    for key_s in [*map_s]:
        if key_s in [*map_t]:
            product += map_s[key_s] * map_t[key_s]
    return product

def generate_map(string, p):
    uniques = {}
    for i in range(len(string) - p + 1):
        if string[i:i+p] not in uniques.keys():
            uniques[string[i:i+p]] = 1
        else:
            uniques[string[i:i+p]] += 1
    return uniques

def kernel2(s, t, p):
    common = 0
    # Record uniques
    unique_s = []
    unique_t = []
    for i in range(len(s) - p + 1):
        unique_s.append(s[i:i+p])
    for j in range(len(t) - p + 1):
        unique_t.append(t[i:i+p])
    unique_s = set(unique_s)
    unique_t = set(unique_t)
    for key_s in unique_s:
        if key_s in unique_t:
            common += 1
    return common


class Perceptron():
    # Instead of w, keep track of index of x, y that made errors
    delta = []
    # store p as a instance
    p = -1
    
    def fit(self, training_data, p):
        self.p = p
        # One pass only
        count = 0
        switch = True
        for i in range(len(training_data)):
            # progress report
            if count > len(training_data)/2 and switch:
                print("Half way done")
                switch = False
            count += 1
            data = training_data[i]
            # Edge case:
            if len(self.delta) == 0:
                self.delta.append(data)
                continue
            # Update case:
            if data[1] * self.w_dot_phi(data) <= 0:
                self.delta.append(data)
        print("All done")
            
    '''
    Helper method which calculates <w_t, phi(x)>
    '''
    def w_dot_phi(self, x):
        # Iterate through all x in list, compute sum of kernals
        dot_product = 0
        for error_data in self.delta:
            dot_product += error_data[1] * kernel(error_data[0], x[0], self.p)
        return dot_product
    
    '''
    Functions for Prediction/Testing
    '''
    # Calculate Error
    def error(self, testing_data):
        # Depending on what method is, calculate predictions
        predictions = self.predict_pass(testing_data)
        return np.mean(predictions != testing_data[:,1])
    
    def predict_pass_one(self, a_test_data):
        if self.w_dot_phi(a_test_data) >= 0:
            return 1
        else:
            return -1
    
    def predict_pass(self, testing_data):
        return np.apply_along_axis(self.predict_pass_one, 1, testing_data[:,:-1])