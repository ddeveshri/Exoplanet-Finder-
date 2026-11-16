"""Run the whole pipeline:  python main.py"""
from src.data import make_splits
from src import train, evaluate

if __name__ == "__main__":
    make_splits()
    train.main()
    evaluate.main()
