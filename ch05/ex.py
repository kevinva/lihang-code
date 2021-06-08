import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from decision_tree import *

df = pd.read_csv('mdata_5-1.txt', index_col=0)
# print(df.head())

cols = df.columns.tolist()
X = df[cols[:-1]].values
y = df[cols[-1]].values
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = DecisionTree(eps=0.02, feas=cols, criterion='gr')
clf.fit(X_train, y_train)
# print(clf.tree)

rst = clf.describe_tree(clf.tree)
# print(rst)
clf.plot_tree(depth=5)

# [
#     { 
#         'name': '有自己的房子', 
#         'value': '10', 
#         'children': [
#             {
#                 'name': '是', 
#                 'value': '5', 
#                 'children': [
#                     {
#                         'name': '批准', 
#                         'value': 10
#                     }
#                 ]
#             }, 
#             {
#                 'name': '否', 
#                 'value': '5', 
#                 'children': [
#                     {
#                         'name': '有工作', 
#                         'value': '5', 
#                         'children': [
#                             {
#                                 'name': '是', 
#                                 'value': '1', 
#                                 'children': [
#                                     {
#                                         'name': '批准', 
#                                         'value': 10
#                                     }
#                                 ]
#                             }, 
#                             {
#                                 'name': '否', 
#                                 'value': '4', 
#                                 'children': [
#                                     {
#                                         'name': '拒绝', 
#                                         'value': 10
#                                     }
#                                 ]
#                             }
#                         ]
#                     }
#                 ]
#             }
#         ]
#     }
# ]