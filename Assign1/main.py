import argparse

MAX_HEIGHT=None
DATA_PATH=None

if __name__=='main':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--data_path",type=str, default="diabetes.csv")

    args = parser.parse_args()
    print(args)
    MAX_HEIGHT = args.max_depth
    DATA_PATH = args.data_path
    if ((MAX_HEIGHT is None) or (MAX_HEIGHT < 0)):
        MAX_HEIGHT = 10

