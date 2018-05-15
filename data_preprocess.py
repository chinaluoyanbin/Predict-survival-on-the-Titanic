# coding: utf-8
import pandas as pd
import os
import numpy as np


def train_preprocess():
    # 读取train.csv为pandas.DataFrame
    train = pd.read_csv(os.getcwd() + '\\data\\train.csv')

    # 数据预处理
    train['Age'] = train['Age'].fillna(train['Age'].mean())
    train['Cabin'] = pd.factorize(train.Cabin)[0]
    train.fillna(0, inplace=True)
    train['Sex'] = [1 if i == 'male' else 0 for i in train.Sex]
    # 处理Pclass
    train['P1'] = np.array(train['Pclass'] == 1).astype(np.int32)
    train['P2'] = np.array(train['Pclass'] == 2).astype(np.int32)
    train['P3'] = np.array(train['Pclass'] == 3).astype(np.int32)
    del train['Pclass']
    # 处理Embarked
    train['E1'] = np.array(train['Embarked'] == 'S').astype(np.int32)
    train['E2'] = np.array(train['Embarked'] == 'C').astype(np.int32)
    train['E3'] = np.array(train['Embarked'] == 'Q').astype(np.int32)
    del train['Embarked']

    # 得到train_x, train_y_
    train_x = train[[
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'P1', 'P2',
        'P3', 'E1', 'E2', 'E3'
    ]]
    train_y_ = train['Survived'].values.reshape(len(train), 1)

    return train_x, train_y_


def test_preproces():
    # 读取test.csv为pandas.DataFrame
    test = pd.read_csv(os.getcwd() + '\\data\\test.csv')

    # 数据预处理
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    test['Cabin'] = pd.factorize(test.Cabin)[0]
    test.fillna(0, inplace=True)
    test['Sex'] = [1 if i == 'male' else 0 for i in test.Sex]
    # 处理Pclass
    test['P1'] = np.array(test['Pclass'] == 1).astype(np.int32)
    test['P2'] = np.array(test['Pclass'] == 2).astype(np.int32)
    test['P3'] = np.array(test['Pclass'] == 3).astype(np.int32)
    del test['Pclass']
    # 处理Embarked
    test['E1'] = np.array(test['Embarked'] == 'S').astype(np.int32)
    test['E2'] = np.array(test['Embarked'] == 'C').astype(np.int32)
    test['E3'] = np.array(test['Embarked'] == 'Q').astype(np.int32)
    del test['Embarked']

    # 得到test_x
    test_x = test[[
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'P1', 'P2',
        'P3', 'E1', 'E2', 'E3'
    ]]

    return test_x
