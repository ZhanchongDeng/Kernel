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
    count = 0
    used_str = []
    # Iterate through s
    for i in range(len(s) - p + 1):
        cur_sub = s[i:i+p] # extract substring
        sub_count = 0
        # Search this in t
        loc = t.find(cur_sub)
        # If existed and not used yet
        if loc >= 0 and cur_sub not in used_str:
            # record
            used_str.append(cur_sub)
            # Get number of substring occured in s and t
            a = count_sub(s, cur_sub)
            b = count_sub(t, cur_sub)
            count += a * b
    return count

def count_sub(string, sub_string):
    count = 0 # keeps track of how many occurances
    cur_index = 0 # pointer
    while cur_index <= len(string)-len(sub_string):
        if string[cur_index:cur_index + len(sub_string)] == sub_string:
            count += 1
        cur_index += 1
    return count



class Perceptron():
    # Instead of w, keep track of which x,y we made error on
    delta = []
    # store p as a instance
    p = -1
    
    def fit(self, training_data, p):
        self.p = p
        # One pass only
        count = 0
        for data in training_data:
            print(count, end=" ")
            count += 1
            # Update case:
            if data[1] * self.w_dot_phi(data) <= 0:
                self.delta.append(data)
            
    '''
    Helper method which calculates <w_t, phi(x)>
    '''
    def w_dot_phi(self, x):
        # Edge case: initial when it's empty
        if len(self.delta) == 0:
            return 0
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