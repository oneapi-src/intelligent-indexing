# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Process raw data download for benchmarks.
"""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    """Read raw dataset and split to train/test csvs for benchmarking.
    """
    if not os.path.exists('News_Category_Dataset_v3.json'):
        print("Please download the dataset and save in this directory!")
        return

    json_list = []
    with open('News_Category_Dataset_v3.json', encoding="utf8") as file:
        for line in file.readlines():
            json_list.append(json.loads(line))
    data = pd.DataFrame.from_records(json_list)

    train, test = train_test_split(data, test_size=0.15, random_state=0)

    if not os.path.exists("huffpost"):
        os.mkdir("huffpost")

    train.to_csv(os.path.join("huffpost", "train_all.csv"), index=False)
    test.to_csv(os.path.join("huffpost", "test.csv"), index=False)


if __name__ == "__main__":
    main()
