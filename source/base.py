from copy import copy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from node import Node


def get_entropy(s: pd.DataFrame) -> float:
    """
    求信息熵

    :param s: 数据集
    :return: 信息熵
    """
    num = len(s)
    if num == 0:
        return 0
    p1 = len(s[s['类别'] == True]) / num
    p2 = len(s[s['类别'] == False]) / num
    return -(p1 * (np.log10(p1) if p1 != 0 else 0) + p2 * (np.log10(p2) if p2 != 0 else 0))


def get_gain(s: pd.DataFrame, label: list) -> float:
    """
    获取信息增益

    :param s: 数据集
    :param label: 特定标签
    :return: 特定标签下的信息增益
    """
    num = len(s)
    es = get_entropy(s)
    esv = 0
    i = 0
    while i < label[1]:
        temp = s[s[label[0]] == i]
        esv += get_entropy(temp) * len(temp) / num
        i += 1
    return es - esv


def get_split(s: pd.DataFrame, label: list) -> float:
    """
    获取分割信息量的度量

    :param s: 数据集
    :param label: 特征属性
    :return: 特定特征属性下的信息分割度量
    """
    num = len(s)
    result = 0
    i = 0
    while i < label[1]:
        temp = s[s[label[0]] == i]
        p = len(temp) / num
        result -= p * np.log2(p) if (p != 0) else 0
        i += 1
    return result


def get_gain_ratio(s: pd.DataFrame, label: list) -> float:
    """
    获取信息增益率， 为信息增益率与分割信息量的度量

    :param s: 数据集
    :param label: 特定标签
    :return: 特定特征属性下的信息增益率
    """
    split = get_split(s, label)
    gain_ration = (get_gain(s, label) / split) if (split != 0) else 0
    return gain_ration


def import_data_set(direction: str, ratio: float = 0.33) -> [list[str, int], pd.DataFrame, pd.DataFrame]:
    """
    导入数据文件（excel），并转换格式，输出标签与数据集

    :param ratio: 数据集与测试集的比例
    :param direction: 文件路径
    :return: [特征属性集，训练集，测试集]
    """
    lbs = [['年龄', 3], ['有工作', 2], ['有自己的房子', 2], ['信贷情况', 3]]
    df = pd.read_excel(direction, 'Sheet1', index_col=0)
    df['年龄'].replace('青年', 0, inplace=True)
    df['年龄'].replace('中年', 1, inplace=True)
    df['年龄'].replace('老年', 2, inplace=True)
    df['有工作'].replace('是', 1, inplace=True)
    df['有工作'].replace('否', 0, inplace=True)
    df['有自己的房子'].replace('是', 1, inplace=True)
    df['有自己的房子'].replace('否', 0, inplace=True)
    df['信贷情况'].replace('一般', 0, inplace=True)
    df['信贷情况'].replace('好', 1, inplace=True)
    df['信贷情况'].replace('非常好', 2, inplace=True)
    df['类别'].replace('是', True, inplace=True)
    df['类别'].replace('否', False, inplace=True)
    train, test = train_test_split(df, test_size=ratio)
    return [lbs, train, test]


def create_tree(current_node: Node,
                s: pd.DataFrame, labels: list[list],
                func: Callable[[pd.DataFrame, list], float]):
    """
    训练、建立决策树

    :param current_node: 根节点
    :param s: 数据集
    :param labels: 特征属性集
    :param func: 建树依据的函数，C4.5导入信息增益率，ID3导入信息增益
    """
    # 无剩余特征属性或数据，抛出异常
    if len(labels) == 0 or len(s) == 0:
        raise Exception("Error: 构造决策树异常，特征属性无法构造决策树")

    # 求取有最大信息增益的特征属性
    max_g = 0
    max_label = labels[0]
    for lb in labels:
        g = func(s, lb)
        if g > max_g:
            max_g = g
            max_label = lb

    # 剩余特征属性无须分类，该节点成为分类结果
    if max_g == 0:
        current_node.label = ['类别', s['类别'].iloc[0]]
        # 此时 current_node.label[0] 为决策结果
        return

    current_node.label = copy(max_label)

    # 用剩余特征属性建立子节点
    i = 0
    while i < max_label[1]:
        child = Node(current_node, i, current_node.level + 1)
        current_node.children.append(child)
        next_labels = copy(labels)
        next_labels.remove(max_label)
        create_tree(child, s[s[max_label[0]] == i], next_labels, func)
        i += 1


def decision(root: Node, s: pd.DataFrame) -> bool:
    """
    依据决策树决策

    :param root: 决策树根节点
    :param s: 待决策数据
    :return:返回决策结果True or False
    """
    current = root
    while current.label[0] != '类别':
        i = s[current.label[0]]
        for node in current.children:
            if node.pindex == i:
                current = node
                break

    return current.label[1]


def get_accuracy_score(pred: list, true: list):
    return accuracy_score(true, pred)


def print_tree(node, indent: list, final_node=True):
    """
    打印树状结构

    :param node: 根节点
    :param indent: 之前部分的信息
    :param final_node: 当前节点是否是父节点最后的孩子节点
    """
    space = '    '
    branch = '│   '
    tee = '├───'
    last = '└───'
    for i in range(node.level):
        print(indent[i], end='')
    if final_node is True:
        print(last, end='')
    else:
        print(tee, end='')
    print(node.to_string())

    cnt = len(node.children)
    for i, n in enumerate(node.children):
        c = space if (final_node is True) else branch
        indent.append(c)
        last_node = (i == cnt - 1)
        print_tree(n, indent, last_node)
        del indent[-1]
