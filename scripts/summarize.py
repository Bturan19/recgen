import csv
from collections import defaultdict

import pandas as pd


def main():
    path = "experiments/results/results.csv"
    df = pd.read_csv(path)
    df = df.groupby("name", as_index=False).last()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
