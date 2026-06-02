import pandas as pd


def load_train(path: str):
    return pd.read_csv(path)


def load_test(path: str):
    return pd.read_csv(path)