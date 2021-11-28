#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#CART_binning
class CART:
    def __init__(self, criterion='gini', splitter='best', max_depth=None, 
                       min_samples_leaf=1000, 
                       max_leaf_nodes=None, 
                       min_impurity_decrease=0, ccp_alpha=0, n_jobs=-1): 
        self.criterion=criterion
        self.splitter=splitter
        self.max_depth=max_depth 
        self.min_samples_leaf=min_samples_leaf 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs
        self.dtree = DecisionTreeClassifier(criterion=self.criterion, 
                                     splitter=self.splitter, 
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_leaf_nodes = self.max_leaf_nodes, 
                                     min_impurity_decrease = self.min_impurity_decrease, 
                                     ccp_alpha = self.ccp_alpha)
    
    def fit(self, x, y):
        '''fits the tree '''

        #with parallel_backend('threading', n_jobs=self.n_jobs):
        self.dtree.fit(x, y)
        return self.dtree

    def get_tree_splits(self):
        '''
        Returns list of thresholds of the deision tree.

        Parameters
        ---------------
        tree: DecisionTreeClassifier object

        Returns
        ---------------
        boundaries: list of thresholds
        '''
        children_left = self.dtree.tree_.children_left
        children_right = self.dtree.tree_.children_right
        threshold = self.dtree.tree_.threshold
        # boundaries of split
        self.boundaries = []

        for i in range(self.dtree.tree_.node_count):
            if children_left[i] != children_right[i]:
                self.boundaries.append(threshold[i])

        return sorted(self.boundaries)
    
    def print_param(self):
        print(self.criterion,
        self.splitter,
        self.max_depth, 
        self.min_samples_leaf, 
        self.max_leaf_nodes, 
        self.min_impurity_decrease,
        self.ccp_alpha)
    

