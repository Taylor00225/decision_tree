import sys

import pandas as pd
from node import Node
import base


if __name__ == "__main__":
    dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame
    labels, dataframe, test_dataframe = base.import_data_set('./data.xlsx')

    id3root = Node(None, -1)
    c45root = Node(None, -1)
    try:
        base.create_tree(id3root, dataframe, labels, base.get_gain)
        print("ID3")
        base.print_tree(id3root, [])
    except Exception as err:
        print("ID3 " + str(err))
        sys.exit(1)
    try:
        base.create_tree(c45root, dataframe, labels, base.get_gain_ratio)
        print("C4.5")
        base.print_tree(c45root, [])
    except Exception as err:
        print("C4.5 " + str(err))
        sys.exit(2)

    i = 0
    n = len(test_dataframe)
    data_id3_pred = []
    data_c45_pred = []
    data_true = []
    while i < n:
        print("*******************************")
        data = test_dataframe.iloc[i]
        data_true.append(data['类别'])
        print(data, end='\n\n')

        print("ID3:", end='')
        pred = base.decision(id3root, data)
        data_id3_pred.append(pred)
        print(pred)

        print("C45:", end='')
        pred = base.decision(c45root, data)
        data_c45_pred.append(pred)
        print(pred, end='\n')
        i += 1

    print("*******************************")
    print("ID3 accuracy: ", end='')
    print(base.get_accuracy_score(data_id3_pred, data_true))
    print("C4.5 accuracy: ", end='')
    print(base.get_accuracy_score(data_c45_pred, data_true))
