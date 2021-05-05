import csv
import numpy as np  
import ast
from datetime import datetime
from math import log, floor, ceil
import random



# to run ensure that the data path is correct ("pid.csv" in the same directory in this case), 
# and envoke the run() function (last line of the code). This example uses the pima indians diabetes 
# dataset

data_path = "pid.csv"
max_tree_depth = 10;
n_trees = 20;


class Utility(object):



    # This method computes entropy for information gain
    def entropy(self, class_y):
        # Input:
        #   class_y         : list of class labels (0's and 1's)

        entropy = 0

        len_class = len(class_y)
        count = {}
        for x in class_y:
            if (x in count):
                count[x] += 1
            else:
                count[x] = 1

        if len(count)>1:
                prob_a = count[0]/len_class
                prob_b = count[1]/len_class
                entropy = (-prob_a*log(prob_a,2))-(prob_b*log(prob_b,2))


        return entropy


    def partition_classes(self, X, y, split_attribute, split_val):
        # Inputs:
        #   X               : data containing all attributes
        #   y               : labels
        #   split_attribute : column index of the attribute to split on
        #   split_val       : a numerical value to divide the split_attribute

        X_left = []
        X_right = []

        y_left = []
        y_right = []
   
        for i in range(len(X)):
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        return (X_left, X_right, y_left, y_right)


    def information_gain(self, previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value

 

        info_gain = 0
        ### Implement your code here
        #############################################
        h = self.entropy(previous_y)
        h_left = self.entropy(current_y[0])
        h_right = self.entropy(current_y[1])

        p_left = len(current_y[0])/(len(current_y[0])+len(current_y[1]))
        p_right = len(current_y[1])/(len(current_y[0])+len(current_y[1]))

        info_gain = h - (h_left*p_left + h_right*p_right)

        return info_gain


    def best_split(self, X, y):
        # Inputs:
        #   X       : Data containing all attributes
        #   y       : labels
        
        split_attribute = 0
        split_val = 0
        X_left, X_right, y_left, y_right = [], [], [], []
        
        results = {'split_attribute':0, 'split_val':0, 'X_left':[], 'X_right':[], 'y_left':[], 'y_right':[], 'info_gain':0}

        n_features_random = round(len(X[0])**.5) # approximatly 8^0.5
        att_list = np.arange(0,len(X[0]),1)
        random_subset = np.random.choice(att_list,n_features_random,replace=False)

        features = [[] for i in range(len(X))]
        for i in range(len(X)):
            for j in random_subset:
                features[i].append(X[i][j])

        for i in range(len(features)):
            for j in range(len(features[0])):
                X_left, X_right, y_left, y_right = self.partition_classes(X, y, random_subset[j], split_val = features[i][j])
                info_gain = self.information_gain(y, [y_left,y_right])

                if info_gain >= results['info_gain']:
                    results['split_attribute'] = random_subset[j]
                    results['split_val'] = features[i][j]
                    results['X_left'] = X_left
                    results['X_right'] = X_right
                    results['y_left'] = y_left
                    results['y_right'] = y_right
                    results['info_gain'] = info_gain

        return results
        

class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth



    def learn(self, X, y, par_node = {}, depth=0):

        ut = Utility()
        self.tree = ut.best_split(X, y)
        self.split(self.tree, max_depth = self.max_depth, depth =0)


        return self.tree

    # sections of this code adapted from https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
    #calc depth of the the dictionary
    def depth_calc(self, node,  level = 1):
        if not isinstance(node, dict) or not node:
            return level
        return max(self.depth_calc(node[key], level + 1) for key in node)


    # this determines the classification at a leaf
    def terminal_node(self, y):
        return(np.argmax(np.bincount(y)))


    # creats tree
    def split(self, node, max_depth, depth):

        X_left = node['X_left']
        X_right = node['X_right']

        y_left = node['y_left']
        y_right = node['y_right']

        # check for no split
        if not y_left or not y_right:
            node['left'] = node['right'] = np.random.randint(0,2)
            return

        # check for minimum value of 1 in branch
        if len(y_left) <2 or len(y_right) <2:
            node['left'] , node['right'] = self.terminal_node(y_left), self.terminal_node(y_right)
            return

        # check for max depth
        if self.depth_calc(self.tree)  >= (self.max_depth):
            node['left'], node['right'] = self.terminal_node(y_left), self.terminal_node(y_right)
            return

        # left branch
        if len(np.unique(y_left)) < 2:
            node['left'] = self.terminal_node(y_left)

        else:
            node['left'] = Utility().best_split(X_left, y_left)
            self.split(node['left'], depth = depth+1, max_depth =  max_depth)

        # right branch
        if len(np.unique(y_right)) < 2:
            node['right'] = self.terminal_node(y_right)
        else:
            node['right'] = Utility().best_split(X_right, y_right)
            self.split(node['right'], depth = depth+1 ,max_depth =  max_depth)




    def classify(self, record):

        def class2(node,record):
            if record[node['split_attribute']] <= node['split_val']:
                if isinstance(node['left'], dict):
                    return class2(node['left'], record)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return class2(node['right'], record)
                else:
                    return node['right']

        result = class2(self.tree,record)

        return(result)




class RandomForest(object):
    num_trees = 0
    decision_trees = []
    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=max_tree_depth) for i in range(num_trees)]
        self.bootstraps_datasets = []
        self.bootstraps_labels = []

    def _bootstrapping(self, XX, n):


        sample = [] # sampled dataset
        labels = []  # class labels for the sampled records

        len_samp = len(XX[0]) - 1
        temp_XX = [XX[x] for x in np.random.choice(range(n), size = n)]


        sample = [x[0:len_samp] for x in temp_XX]
        labels = [x[-1] for x in temp_XX]

        return (sample, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):

        for i in range(0,self.num_trees):
            X = self.bootstraps_datasets[i]
            y = self.bootstraps_labels[i]
            self.decision_trees[i].learn(X,y)


    def voting(self, X):
        y = []
        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)

            if len(counts) == 0:
                counts = np.bincount([1])
                y = np.append(y, np.argmax(counts))
            else:
                 y = np.append(y, np.argmax(counts))

        y = [int(x) for x in y]

        return y

    def user(self):
        """
        :return: string
        """
        return 'randomString'

def get_forest_size():
    forest_size = n_trees
    return forest_size

def get_random_seed():
    random_seed = 0
    return random_seed
    
def run():
    np.random.seed(get_random_seed())
    # start time 
    start = datetime.now()
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([i for i in range(0, 9)])  # indices of numeric attributes (columns)

    # Loading data set
    print("reading the data")
    with open(data_path) as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])
            

    # Initializing a random forest.
    randomForest = RandomForest(get_forest_size())

    # printing the name
    #print("__Name: " + randomForest.user()+"__")

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))

    # end time
    print("Execution time: " + str(datetime.now() - start))
    
run()