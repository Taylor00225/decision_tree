from typing import Any


class Node:
    label: list | None  # 特征属性
    pindex: int     # 父节点的特征属性下的分类，用数字替代文字，如0代表否，1代表是
    parent: Any     # 父节点
    children: list[Any]
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
        if self.label[0] == '类别':
            string += ' ' + str(self.label[1])
        return string
