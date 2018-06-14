# coding: utf-8
import pandas as pd
import numpy as np


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def train_preprocess():
    # 读取train.csv为pandas.DataFrame
    train = pd.read_csv('../input/train.csv')

    # Pclass
    # One-hot编码
    train['P1'] = np.array(train['Pclass'] == 1).astype(np.int32)
    train['P2'] = np.array(train['Pclass'] == 2).astype(np.int32)
    train['P3'] = np.array(train['Pclass'] == 3).astype(np.int32)

    # Sex
    # 把male/female转换成1/0
    train['Sex'] = [1 if i == 'male' else 0 for i in train.Sex]

    # SibSp and Parch
    # 'FamilySize'：家庭成员人数
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    # 'IsAlone'：是否只身一人
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

    # Embarked
    train['Embarked'] = train['Embarked'].fillna('S')
    # One-hot编码
    train['E1'] = np.array(train['Embarked'] == 'S').astype(np.int32)
    train['E2'] = np.array(train['Embarked'] == 'C').astype(np.int32)
    train['E3'] = np.array(train['Embarked'] == 'Q').astype(np.int32)

    # Fare
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    train['CategoricalFare'].cat.categories = [1, 2, 3, 4]
    # one-hot编码
    train['F1'] = np.array(train['CategoricalFare'] == 1).astype(np.int32)
    train['F2'] = np.array(train['CategoricalFare'] == 2).astype(np.int32)
    train['F3'] = np.array(train['CategoricalFare'] == 3).astype(np.int32)
    train['F4'] = np.array(train['CategoricalFare'] == 4).astype(np.int32)

    # Age
    age_avg = train['Age'].mean()
    age_std = train['Age'].std()
    age_null_count = train['Age'].isnull().sum()
    age_null_random_list = np.random.randint(
        age_avg - age_std, age_avg + age_std, size=age_null_count)
    train['Age'][np.isnan(train['Age'])] = age_null_random_list
    train['Age'] = train['Age'].astype(int)
    train['CategoricalAge'] = pd.qcut(train['Age'], 5)
    train['CategoricalAge'].cat.categories = [1, 2, 3, 4, 5]
    train['A1'] = np.array(train['CategoricalAge'] == 1).astype(np.int32)
    train['A2'] = np.array(train['CategoricalAge'] == 2).astype(np.int32)
    train['A3'] = np.array(train['CategoricalAge'] == 3).astype(np.int32)
    train['A4'] = np.array(train['CategoricalAge'] == 4).astype(np.int32)
    train['A5'] = np.array(train['CategoricalAge'] == 5).astype(np.int32)

    # Name
    train['Title'] = train['Name'].apply(get_title)
    train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train['Title'] = train['Title'].replace('Mlle', 'Miss')
    train['Title'] = train['Title'].replace('Ms', 'Miss')
    train['Title'] = train['Title'].replace('Mme', 'Mrs')
    train['T1'] = np.array(train['Title'] == 'Master').astype(np.int32)
    train['T2'] = np.array(train['Title'] == 'Miss').astype(np.int32)
    train['T3'] = np.array(train['Title'] == 'Mr').astype(np.int32)
    train['T4'] = np.array(train['Title'] == 'Mrs').astype(np.int32)
    train['T5'] = np.array(train['Title'] == 'Rare').astype(np.int32)

    # 数据清洗
    train_x = train[[
        'P1', 'P2', 'P3', 'Sex', 'IsAlone', 'E1', 'E2', 'E3', 'F1', 'F2', 'F3',
        'F4', 'A1', 'A2', 'A3', 'A4', 'A5', 'T1', 'T2', 'T3', 'T4', 'T5'
    ]]
    train_y_ = train[['Survived']]

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