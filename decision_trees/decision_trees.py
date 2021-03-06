# -*- coding: utf-8 -*-


from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

class Tree:
    '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.
    Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
    '''

    def leaf(data):
        '''Create a leaf tree
        '''
        return Tree(data=data)

    # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right)

    # all arguments after `*` are *keyword-only*!
    def __init__(self, *, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        '''Check if this tree is a leaf tree
        '''
        return self.left == None and self.right == None

    def children(self):
        '''List of child subtrees
        '''
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        '''Compute the depth of a tree
        A leaf is depth-1, and a child is one deeper than the parent.
        '''
        return max([x.depth() for x in self.children()], default=0) + 1

l_nah = Tree.leaf('nah')
l_like = Tree.leaf('like')


Tr_1 = Tree(data = 'likedOtherSys?', left = l_nah, right = l_like)
Tr_2 = Tree(data = 'morning?', left = l_like, right = l_nah)
Tr_3 = Tree(data = 'takenOtherSys?', left = Tr_2, right = Tr_1)
Tr_4 = Tree(data = 'isSystems?', left = l_like, right = Tr_3)
print(Tr_4)

"""Task 2"""

df = pd.read_csv('data.csv')
df = df.assign(ok=lambda df: df.rating >= 0)
print(df, '\n')

"""Task 3"""

def single_feature_score(data, goal, feature):

    true_match = 0
    false_match = 0
    true_items = data[data[feature] == True][goal]
    false_items = data[data[feature] == False][goal]
    
    if len(true_items) != 0:
        true_most_common = Counter(true_items.tolist()).most_common(1)[0][0]
        true_match = true_items.value_counts()[true_most_common]        
    
    if len(false_items) != 0:
        false_most_common = Counter(false_items.tolist()).most_common(1)[0][0]
        false_match = false_items.value_counts()[false_most_common]      
    
    return (true_match + false_match) / len(data)

def best_and_worst_feature(data, goal, features):
    scorer = partial(single_feature_score, data, goal)
    return max(features, key=scorer), min(features, key=scorer)

features = df.drop(['rating', 'ok'], axis=1).columns.tolist()
b, w =  best_and_worst_feature(df, 'ok', features)
b_score = single_feature_score(df, 'ok', b)
w_score = single_feature_score(df, 'ok', w)

print(b_score)
print(w_score)

"""Task 4"""

len(pd.unique([1, 2, 1]))

def DecisionTreeTrain(data, goal, features):
    
    guess = Counter(data[goal].tolist()).most_common(1)[0][0]
    
    if len(features) == 0 or len(pd.unique(data[goal])) == 1:
        return Tree.leaf(guess)
    else:
        b, _ = best_and_worst_feature(data, goal, features)
        remaining_features = features.copy()
        remaining_features.remove(b)        
        true_items = data[data[b] == True]
        false_items = data[data[b] == False]
        
        if len(true_items) != 0:
            right = DecisionTreeTrain(true_items, goal, remaining_features)
        else:
            right = Tree.leaf(guess)
        
        if len(false_items) != 0:
            left = DecisionTreeTrain(false_items, goal, remaining_features)
        else:
            left = Tree.leaf(guess)         
        
        return Tree(data=b, left=left, right=right)
    
def DecisionTreeTest(tree, data_point):
    if tree.is_leaf():
        return tree.data
    else:
        feature_name = tree.data
        if not data_point[feature_name]:
            return DecisionTreeTest(tree.left, data_point)
        else:
            return DecisionTreeTest(tree.right, data_point)

tr = DecisionTreeTrain(df, 'ok', features)
print(tr)

"""Task 5"""

def DecisionTreeTrainWithMaxDepth(data, goal, features, max_depth):
    
    guess = Counter(data[goal].tolist()).most_common(1)[0][0]
    
    if len(features) == 0 or len(pd.unique(data[goal])) == 1 or max_depth == 0:
        return Tree.leaf(guess)
    else:
        b, _ = best_and_worst_feature(data, goal, features)
        remain_features = features.copy()
        remain_features.remove(b)        
        true_items = data[data[b] == True]
        false_items = data[data[b] == False]
        
        if len(true_items) != 0:
            right = DecisionTreeTrainWithMaxDepth(true_items, goal, remain_features, max_depth-1)
        else:
            right = Tree.leaf(guess)
        
        if len(false_items) != 0:
            left = DecisionTreeTrainWithMaxDepth(false_items, goal, remain_features, max_depth-1)
        else:
            left = Tree.leaf(guess)         
        
        return Tree(data=b, left=left, right=right)

tr2 = DecisionTreeTrainWithMaxDepth(df, 'ok', features, max_depth=3)
print(tr2)

"""Performance"""

def calc_accuracy(tree, goal, data):
    data = data.to_dict(orient='records')
    
    prediction_y = [DecisionTreeTest(tree, point) for point in data]
    true_y = [point[goal] for point in data]    
    N = len(prediction_y)
    prediction_y = np.array(prediction_y)
    true_y = np.array(true_y)
    
    return (true_y == prediction_y).sum() / N

max_depths = [i for i in range(10)]
scores = list()
for i in max_depths:
    tree = DecisionTreeTrainWithMaxDepth(df, 'ok', features, max_depth=i)
    scores.append(calc_accuracy(tree, 'ok', df))
