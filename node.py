import pandas as pd
from typing import Any


class Node:
    label: list | None
    pindex: int
    parent: Any
    children: list[Any]
    data: list[pd.DataFrame] | None
    level: int

    def __init__(self, parent, index, level=0):
        self.label = None
        self.parent = parent
        self.pindex = index
        self.children = []
        self.data = None
        self.level = level

    def to_string(self):
        string = str(self.pindex) + ' ' + self.label[0]
        return string
