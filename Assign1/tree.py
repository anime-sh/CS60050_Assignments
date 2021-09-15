import numpy as np
import utils
from graphviz import Digraph


class Node:
    '''
        Class defines the nodes of the decision tree
        self.attr: [String] attribute of the node
        self.val: [Float] value of the attribute
        self.avg_attr: [Float] average of the attribute

        self.left: [Node] left child of the node
        self.right: [Node] right child of the node

    '''

    def __init__(self, attribute, value, type_arr):
        '''
            Initializes the node
        '''
        self.attr_idx = attribute
        self.val = value
        self.attr_type = type_arr[self.attr_idx]
        self.left = None
        self.right = None
        self.leaf = False
        self.classification = None

    def make_leaf(self, classification):
        '''
            Makes the node a leaf node
        '''
        # print(f'Making leaf node {classification}')
        self.leaf = True
        self.classification = classification

    def get_classification(self):
        '''
            Returns the classification of the node
        '''
        return self.classification

    def predict_node(self, X):
        '''
            Predicts the class of the node
        '''
        if self.leaf:
            return self.classification
        else:
            if self.attr_type == 'cont':
                if X[self.attr_idx] <= self.val:
                    return self.left.predict_node(X)
                else:
                    return self.right.predict_node(X)
            else:
                if X[self.attr_idx] == self.val:
                    return self.left.predict_node(X)
                else:
                    return self.right.predict_node(X)


class DecisionTree:
    '''
        Class defines the decision tree
        self.root: [Node] root of the tree
        self.X:  [Nest List] training features
        self.y:  [List] training labels
    '''

    def __init__(self, X, y, column_names, min_leaf_size, max_depth):
        '''
            Initializes the tree
        '''
        self.root = None
        self.X = X
        self.y = y
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.column_names = column_names
        self.type_arr = utils.assign_feature_type(X, 4)

    def fit(self):
        '''
                Builds the decision tree
        '''
        self.root = self.build_tree(self.X, self.y)

    def is_leaf(self, X_lo, y_lo):
        '''
            Checks if the node is a leaf
            node: [Node] node to be checked
        '''
        if X_lo.shape[0] <= self.min_leaf_size:
            return True
        if utils.check_purity(y_lo):
            return True
        return False

    def build_tree(self, X, y, depth=0):
        '''
            Recursively builds the decision tree
            node: [Node] node to be built
            depth: [Int] depth of the node
        '''

        node = Node(0, 0, self.type_arr)
        if self.is_leaf(X, y) or depth == self.max_depth:
            # print(np.unique(y,return_counts=True))
            node.make_leaf(utils.classify_array(y))
            return node
        else:
            depth += 1
            best_attr, best_val = utils.get_best_split(
                X, y, self.type_arr)
            # print(best_attr,best_val,self.type_arr[best_attr])
            node.attr_idx = best_attr
            node.val = best_val
            node.attr_type = self.type_arr[best_attr]
            X_left, y_left, X_right, y_right = utils.create_children_np(
                X, y, best_attr, best_val, self.type_arr)
            # print(X_left.shape,y_left.shape,X_right.shape,y_right.shape)
            left_tree = self.build_tree(X_left, y_left, depth)
            right_tree = self.build_tree(X_right, y_right, depth)
            if left_tree == right_tree:
                # print(f"DEPTH = {depth} left is right")
                node.make_leaf(utils.classify_array(y_left))
            else:
                node.leaf = False
                node.left = left_tree
                node.right = right_tree
            return node

    def predict(self, X):
        '''
            Predicts the labels of the test data
            X: [Nest List] test features
        '''
        if self.root == None:
            return None
        else:
            return np.array([self.root.predict_node(x) for x in X])

    def calc_accuracy(self, X, y):
        '''
            Calculates the accuracy of the decision tree
            X: [Nest List] test features
            y: [List] test labels
        '''
        y_pred = self.predict(X)
        return utils.calc_accuracy(y, y_pred)

    def print_tree(self, node=None, depth=0):
        '''
            Prints the tree in a readable format
            node: [Node] node to be printed
            depth: [Int] depth of the node
        '''

        if node.leaf:
            print('|'*depth+'Leaf: '+str(node.classification))
        else:
            compoperator = '<='
            if node.attr_type == 'discrete':
                compoperator = '=='
            print('|'*depth+'Attribute: ' +
                  self.column_names[node.attr_idx]+'  ' + compoperator + '  Value: '+str(node.val))
            self.print_tree(node.left, depth+1)
            self.print_tree(node.right, depth+1)


def render_node(vertex, feature_names):
    if vertex.leaf:
        return f'{vertex.classification}\n'
    return f'{feature_names[vertex.attr_idx]} <= {vertex.val})\n'


def tree_to_gv(node_root, feature_names):
    f = Digraph('Decision Tree', filename='decision_tree.gv')
    f.attr(rankdir='LR', size='1000,500')
    f.attr('node', shape='rectangle')
    q = [node_root]
    while len(q) > 0:
        node = q.pop(0)
        if node.left != None:
            f.edge(render_node(node, feature_names), render_node(
                node.left, feature_names), label='True')
        if node.right != None:
            f.edge(render_node(node, feature_names), render_node(
                node.right, feature_names), label='False')

        if node.left != None:
            q.append(node.left)
        if node.right != None:
            q.append(node.right)

    f.render('./decision_tree.gv', view=True)
