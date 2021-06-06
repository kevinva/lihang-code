
from math import sqrt
from collections import namedtuple

class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right


class KdTree:
    def __init__(self, data):
        k = len(data[0])

        def createNode(split, data_set):
            if not data_set:
                return None
            
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KdNode(
                median,
                split,
                createNode(split_next, data_set[:split_pos]),
                createNode(split_next, data_set[split_pos + 1:])
            )

        self.root = createNode(0, data)


def preorder(root):
    print(root.dom_elt)

    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)



def find_nearest(tree, point):
    k = len(point)
    result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1
        s = kd_node.split
        pivot = kd_node.dom_elt

        if target[s] <= pivot[s]:  # 目标离左子树更近
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left
            
        temp1 = travel(nearer_node, target, max_dist) # 进行遍历找到包含目标点的区域
        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist
        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内


        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:   # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断
            #（这里好像有问题：与叶节点的超平面不相交不代表与叶节点的父节点的超平面不相交）

        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        temp2 = travel(further_node, target, max_dist)
        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))

        