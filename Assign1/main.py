import argparse
import utils
import tree
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
MAX_HEIGHT = None
DATA_PATH = None
MIN_LEAF_SIZE = None
MEASURE = None
RANDOM_SEED = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_leaf_size", type=int, default=1)
    parser.add_argument("--impurity_measure", type=str, default="entropy")
    parser.add_argument("--random_seed", type=int, default="42")
    parser.add_argument("--data_path", type=str, default="diabetes.csv")

    args = parser.parse_args()
    
    MAX_HEIGHT = args.max_depth
    DATA_PATH = args.data_path
    MIN_LEAF_SIZE = args.min_leaf_size
    MEASURE = args.impurity_measure
    RANDOM_SEED = args.random_seed

    print("f MAX DEPTH = {MAX_HEIGHT}")
    print("f DATA PATH = {DATA_PATH}")
    
    if (MAX_HEIGHT < 0):
        MAX_HEIGHT = 10

    print("-"*50 + "PREPROCESSING" + "-"*50)
    feature_names = utils.get_column_names(DATA_PATH)
    X_full, y_full = utils.get_X_y(DATA_PATH)
    X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("\n"+"-"*50 + "TREE with Gini" + "-"*50)
    MEASURE="gini"
    Tree_1_gini = tree.DecisionTree(
        X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)

    Tree_1_gini.fit()
    print("Training complete")
    print("Training accuracy:", Tree_1_gini.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", Tree_1_gini.calc_accuracy(X_test, y_test))

    print("\n"+"-"*50 + "TREE with Entropy" + "-"*50)
    MEASURE="entropy"
    Tree_1_entropy = tree.DecisionTree(
        X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)

    Tree_1_entropy.fit()
    print("Training complete")
    print("Training accuracy:", Tree_1_entropy.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", Tree_1_entropy.calc_accuracy(X_test, y_test))

    print("\n"+"-"*50 + "TREE over 10 random splits GINI" + "-"*50)
    MEASURE="gini"
    BEST_TREE_GINI=Tree_1_gini
    for i in range(1,11):
        X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8,seed=i)
        Tree_2_gini=tree.DecisionTree(
            X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)
        Tree_2_gini.fit()
        print(f"Training of {i} tree complete")
        print(f"Training accuracy: {Tree_2_gini.calc_accuracy(X_train, y_train,print_report=False)}")
        print(f"Testing accuracy: {Tree_2_gini.calc_accuracy(X_test, y_test,print_report=False)}")
        if Tree_2_gini.calc_accuracy(X_test, y_test,print_report=False) > BEST_TREE_GINI.calc_accuracy(X_test, y_test,False):
            BEST_TREE_GINI=Tree_2_gini
    
    print("\n"+"-"*50 + "BEST TREE OVER 10 RANDOM SPLITS GINI" + "-"*50)
    print(f"Training accuracy: {BEST_TREE_GINI.calc_accuracy(X_train, y_train)}")
    print(f"Testing accuracy: {BEST_TREE_GINI.calc_accuracy(X_test, y_test)}")
    tree.tree_to_gv(BEST_TREE_GINI.root, feature_names,"unpruned_BEST_GINI.gv")    


    print("\n"+"-"*50 + "TREE over 10 random splits ENTROPY" + "-"*50)
    MEASURE="entropy"
    BEST_TREE_ENTROPY=Tree_1_entropy
    for i in range(1,11):
        X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8,seed=i)
        Tree_2_entropy=tree.DecisionTree(
            X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)
        Tree_2_entropy.fit()
        print(f"Training of {i} tree complete")
        print(f"Training accuracy: {Tree_2_entropy.calc_accuracy(X_train, y_train,print_report=False)}")
        print(f"Testing accuracy: {Tree_2_entropy.calc_accuracy(X_test, y_test,print_report=False)}")
        if Tree_2_entropy.calc_accuracy(X_test, y_test,print_report=False) > BEST_TREE_ENTROPY.calc_accuracy(X_test, y_test,False):
            BEST_TREE_ENTROPY=Tree_2_entropy
        
    print("\n"+"-"*50 + "BEST TREE OVER 10 RANDOM SPLITS ENTROPY" + "-"*50)
    print(f"Training accuracy: {BEST_TREE_ENTROPY.calc_accuracy(X_train, y_train)}")
    print(f"Testing accuracy: {BEST_TREE_ENTROPY.calc_accuracy(X_test, y_test)}")
    tree.tree_to_gv(BEST_TREE_ENTROPY.root, feature_names,"unpruned_BEST_ENTROPY.gv")
    
    print("\n"+"-"*50 + "DEPTH Vs Accuracy" + "-"*50)
    X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8)
    acc_depth_list=[]
    acc_node_list=[]
    for i in range(1,11):
        MAX_HEIGHT=i
        Tree_3=tree.DecisionTree(
            X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)
        Tree_3.fit()
        acc_depth_list.append(Tree_3.calc_accuracy(X_test, y_test,print_report=False))
        acc_node_list.append(Tree_3.count_nodes())
    
    plt.plot(range(1,11),acc_depth_list)
    plt.xlabel("Max Height")
    plt.ylabel("Accuracy")
    plt.title("Depth Vs Accuracy")
    plt.show()

    plt.plot(acc_node_list,acc_depth_list)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.title("Number of Nodes Vs Accuracy")
    plt.show()

    OPTIMAL_DEPTH=1+np.argmax(np.array(acc_depth_list))
    print(f"BEST DEPTH  = { OPTIMAL_DEPTH }")
    
    
    
    
    
    
    
    # X_train, X_val, X_test, y_train, y_val, y_test = utils.train_val_test_split(
    #     X_full, y_full, 0.6, 0.2,seed=RANDOM_SEED)
    # print("X_train:", X_train.shape)
    # print("X_test:", X_test.shape)
    # print("y_train:", y_train.shape)
    # print("y_test:", y_test.shape)
    # print("X_val:", X_val.shape)
    # print("y_val:", y_val.shape)

    # Tree = tree.DecisionTree(
    #     X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)

    # Tree.fit()
    # print("Training complete")
    # print("Training accuracy:", Tree.calc_accuracy(X_train, y_train))
    # print("Testing accuracy:", Tree.calc_accuracy(X_test, y_test))
    # print("Validation accuracy:", Tree.calc_accuracy(X_val, y_val))
    # print("Tree:\n\n\n")
    # # Tree.print_tree(Tree.root)
    # tree.tree_to_gv(Tree.root, feature_names,"unprunedDT.gv")

    # Tree.post_prune(X_train, y_train, X_val, y_val)
    # print("Post Pruning complete")
    # print("Training accuracy:", Tree.calc_pruned_accuracy(X_train, y_train))
    # print("Testing accuracy:", Tree.calc_pruned_accuracy(X_test, y_test))
    # print("Validation accuracy:", Tree.calc_pruned_accuracy(X_val, y_val))
    # tree.tree_to_gv(Tree.root_pruned, feature_names, "prunedDT.gv")
