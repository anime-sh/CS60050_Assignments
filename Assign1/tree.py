
class Node:
    '''
        Class defines the nodes of the decision tree
        self.attr: [String] attribute of the node
        self.val: [Float] value of the attribute
        self.avg_attr: [Float] average of the attribute

        self.left: [Node] left child of the node
        self.right: [Node] right child of the node

    '''
    def __init__(self, attribute, value, average_attribute):
        '''
            Initializes the node
        '''
        self.attr=attribute
        self.val=value
        self.avg_attr=average_attribute
        
        self.left=None
        self.right=None        


class DecisionTree:
    '''
        Class defines the decision tree
        self.root: [Node] root of the tree
        self.X:  [Nest List] training features
        self.y:  [List] training labels
    '''
    

