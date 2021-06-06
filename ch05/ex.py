import numpy as np
from decision_tree import cal_ent

myList = np.array([1, 2, 3, 4, 5, 6, 1])
x_values = list(set(myList))
for x_value in x_values:
    print(myList[myList == x_value])
cal_ent(myList)