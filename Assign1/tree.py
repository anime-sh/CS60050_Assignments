
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

    def reset_node(self,attribute,left, right):
        '''
            Resets the node
        '''
        self.attr=attribute
        self.left=left
        self.right=right
    
    def prune_node(self):
        '''
            Prunes the node
        '''
        self.attr= "Pruned"
        self.left=None
        self.right=None

    def dfs_count(self):
        '''
            Counts the number of nodes in the tree
        '''
        count=1
        if not self.left is None:
            count+=self.left.dfs_count()
        if not self.right is None:
            count+=self.right.dfs_count()
        return count

