import argparse
import utils
import tree
import pandas as pd
import seaborn as sns
MAX_HEIGHT = None
DATA_PATH = None

# if __name__ == 'main':
parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_leaf_size", type=int, default=2)
parser.add_argument("--data_path", type=str, default="diabetes.csv")

args = parser.parse_args()
print(args)
MAX_HEIGHT = args.max_depth
DATA_PATH = args.data_path
MIN_LEAF_SIZE = args.min_leaf_size

if (MAX_HEIGHT < 0):
    MAX_HEIGHT = 10

feature_names = utils.get_column_names(DATA_PATH)
X_full, y_full = utils.get_X_y(DATA_PATH)
X_train, X_test, y_train, y_test = utils.train_test_split(X_full, y_full,0.8)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

tree = tree.DecisionTree(
    X_train, y_train, feature_names, MIN_LEAF_SIZE, MAX_HEIGHT)

tree.fit()
print("Training complete")
print("Training accuracy:", tree.calc_accuracy(X_train, y_train))
print("Testing accuracy:", tree.calc_accuracy(X_test, y_test))

print("Tree:\n\n\n")

tree.print_tree(tree.root)

