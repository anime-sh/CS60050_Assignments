import argparse
import utils
import tree
import pandas as pd
import seaborn as sns
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
    print(args)
    MAX_HEIGHT = args.max_depth
    DATA_PATH = args.data_path
    MIN_LEAF_SIZE = args.min_leaf_size
    MEASURE = args.impurity_measure
    RANDOM_SEED = args.random_seed
    if (MAX_HEIGHT < 0):
        MAX_HEIGHT = 10

    feature_names = utils.get_column_names(DATA_PATH)
    X_full, y_full = utils.get_X_y(DATA_PATH)
    # X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_val_test_split(
        X_full, y_full, 0.6, 0.2,seed=RANDOM_SEED)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    Tree = tree.DecisionTree(
        X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT, MEASURE)

    Tree.fit()
    print("Training complete")
    print("Training accuracy:", Tree.calc_accuracy(X_train, y_train))
    print("Testing accuracy:", Tree.calc_accuracy(X_test, y_test))
    print("Validation accuracy:", Tree.calc_accuracy(X_val, y_val))
    print("Tree:\n\n\n")
    # Tree.print_tree(Tree.root)
    tree.tree_to_gv(Tree.root, feature_names,"unprunedDT.gv")

    Tree.post_prune(X_train, y_train, X_val, y_val)
    print("Post Pruning complete")
    print("Training accuracy:", Tree.calc_pruned_accuracy(X_train, y_train))
    print("Testing accuracy:", Tree.calc_pruned_accuracy(X_test, y_test))
    print("Validation accuracy:", Tree.calc_pruned_accuracy(X_val, y_val))
    tree.tree_to_gv(Tree.root_pruned, feature_names, "prunedDT.gv")
