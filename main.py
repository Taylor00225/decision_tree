import pandas as pd
from node import Node
import base


if __name__ == "__main__":
    dataframe: pd.DataFrame
    labels, dataframe = base.import_data_set('./data.xlsx')

    id3root = Node(None, -1)
    base.create_tree(id3root, dataframe, labels, base.get_gain)

    c45root = Node(None, -1)
    base.create_tree(c45root, dataframe, labels, base.get_gain_ratio)

    data = dataframe.iloc[2]
    print(data)
    print()
    print("ID3:")
    # base.traval_tree(id3root)
    base.print_tree(id3root, [])
    print(str(base.decision(id3root, data)) + "\n\n")
    print("C45:")
    # base.traval_tree(c45root)
    base.print_tree(c45root, [])
    print(base.decision(c45root, data))
