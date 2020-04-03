"""
用pandas实现决策树（ID3算法）
Author: zhs
Date: 2020.4.3
"""
import math
import pandas as pd


def splitSubset(data_df, features, split_feature, value):
    """根据当前决策结点属性取值的不同，划分样本集data"""
    mask = data_df[features[split_feature]] == value
    return data_df[mask]

def shannonEntropy(data_df):
    """计算划分特征之前数据集的香农熵"""
    D = len(data_df.values)
    label_df = data_df['label']
    D_i1 = label_df[label_df=='yes'].count()
    D_i2 = label_df[label_df=='no'].count()
    base_entropy = -D_i1/D*math.log2(D_i1/D) - D_i2/D*math.log2(D_i2/D)

    return base_entropy

def featureEntropy(E_D_A, D, D_i, D_i1, D_i2):
    """计算使用特征值A分割数据集后计算的信息熵"""
    if D_i1 != 0 and D_i2 != 0:
        E_D_A += -D_i / D * (D_i1 / D_i * math.log2(D_i1 / D_i) + D_i2 / D_i * math.log2(D_i2 / D_i))
    elif D_i1 == 0:
        E_D_A += -D_i / D * (D_i2 / D_i * math.log2(D_i2 / D_i))
    elif D_i2 == 0:
        E_D_A += -D_i / D * (D_i1 / D_i * math.log2(D_i1 / D_i))

    return E_D_A

def chooseBestFeature(data_df, features, features_values):
    """选取信息熵增益最大的特征作为划分"""
    D = len(data_df.values)
    base_entropy = shannonEntropy(data_df)
    best_gain = 0
    best_feature = -1

    info_gains = []
    for i, f_v in enumerate(features_values):
        E_D_A = 0
        for v in f_v:
            axis = features[i]
            mask = data_df[axis] == v
            label_df = data_df[mask]['label']
            D_i = label_df.count()
            D_i1 = label_df[label_df == 'yes'].count()
            D_i2 = label_df[label_df == 'no'].count()
            # 计算使用特征值A分割数据集后计算的信息熵
            E_D_A = featureEntropy(E_D_A, D, D_i, D_i1, D_i2)

        if (base_entropy-E_D_A) > best_gain:
            best_gain = base_entropy - E_D_A
            best_feature = i

        info_gains.append(base_entropy-E_D_A)

    # print("信息熵增益列表：", info_gains)
    return best_feature


def createTree(data_df, features, features_values):
    """递归构建决策树"""
    # 递归终止条件，叶结点的样本属于同一类别或者data_df为空
    if len(data_df.values) == 0:
        return
    label_df = data_df['label']
    D_i1 = label_df[label_df == 'yes'].count()
    D_i2 = label_df[label_df == 'no'].count()
    if D_i1==len(label_df.values) or D_i2==len(label_df.values):
        return label_df.values[0]

    # 用=和list产生新变量，防止后面删除列表元素对原变量产生影响
    features_tmp = list(features)
    features_values_tmp = list(features_values)
    # print("features: ", features_tmp)
    # print("feature_values: ", features_values_tmp)

    best_feature = chooseBestFeature(data_df, features_tmp, features_values_tmp)
    best_label = features_tmp[best_feature]
    myTree = {best_label: {}}  # 构建决策树字典

    value_set = features_values_tmp[best_feature]
    features_tmp.pop(best_feature)
    features_values_tmp.pop(best_feature)

    for v in value_set:
        sub_df = splitSubset(data_df, features, best_feature, v)
        myTree[best_label][v] = createTree(data_df=sub_df,
                                           features=features_tmp, features_values=features_values_tmp)

    return myTree


def predict(tree, features, x):
    """预测新数据特征下是否进行活动"""
    for key1 in tree.keys():
        secondDict = tree[key1]
        # key是根节点代表的特征，featIndex是取根节点特征在特征列表的索引，方便后面对输入样本逐变量判断
        featIndex = features.index(key1)
        # 这里每一个key值对应的是根节点特征的不同取值
        for key2 in secondDict.keys():
            # 找到输入样本在决策树中的由根节点往下走的路径
            if x[featIndex] == key2:
                # 该分支产生了一个内部节点，则在决策树中继续同样的操作查找路径
                if type(secondDict[key2]).__name__ == "dict":
                    classLabel = predict(secondDict[key2], features, x)
                # 该分支产生是叶节点，直接取值就得到类别
                else:
                    classLabel = secondDict[key2]

    return classLabel


if __name__ == '__main__':
    data = [
                [2, 2, 1, 0, "yes"],
                [2, 2, 1, 1, "no"],
                [1, 2, 1, 0, "yes"],
                [0, 0, 0, 0, "yes"],
                [0, 0, 0, 1, "no"],
                [1, 0, 0, 1, "yes"],
                [2, 1, 1, 0, "no"],
                [2, 0, 0, 0, "yes"],
                [0, 1, 0, 0, "yes"],
                [2, 1, 0, 1, "yes"],
                [1, 2, 0, 0, "no"],
                [0, 1, 1, 1, "no"],
            ]
    # 分类属性列表
    features = ['weather', 'temperature', 'humidity', 'wind']
    data_df = pd.DataFrame(data, columns=['weather', 'temperature', 'humidity', 'wind', 'label'])

    data_arr = data_df.values[:, :-1]
    features_values = []
    for i in range(len(features)):
        features_set = set(data_arr[:, i])
        features_values.append(features_set)

    # 尝试计算第一轮各特征的信息熵增益
    # chooseBestFeature(data_df, features, features_values)

    my_tree = createTree(data_df, features, features_values)
    print(my_tree)

    label = predict(my_tree, features, [1, 1, 1, 0])
    print("新数据[1,1,1,0]对应的是否要进行活动为:{}".format(label))