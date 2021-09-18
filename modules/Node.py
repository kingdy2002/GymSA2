import math

class Node(object):

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def update_key_and_value(self, new_key, new_value):
        self.update_key(new_key)
        self.update_value(new_value)

    def update_key(self, new_key):
        self.key = new_key

    def update_value(self, new_value):
        self.value = new_value

    def __eq__(self, other):
        return self.key == other.key and self.value == other.value


class MCTsNode(object) :
    def __init__(self,parent,prior_p, c_ratio) :
        self.parent = parent
        self.children = {}
        self.visited_n = 0
        self.W = 0
        self.Q = 0
        self.U = 0
        self.P = prior_p
        self.c_ratio = c_ratio
        self.update_U()

    def update_U(self):
        if self.parent is not None :
            self.U = self.Q + self.c_ratio*self.P*math.sqrt(self.parent.visited_n +1)/(1+self.visited_n)

    def update(self,child_q) :
        self.visited_n += 1
        self.W += child_q
        self.Q = self.W/self.visited_n
        self.update_U()

    def update_all(self,child_q) :
        if self.parent :
            self.parent.update_all(-child_q)
        self.update(child_q)

    def search(self) :
        return max(self.children.items(), key = lambda child : child[1].U)

    def add_chiled(self, action_prob, valid_move) :
        for move in valid_move:
            if move not in self.children :
                self.children[move] = MCTsNode(self,action_prob[0][move].item(),self.c_ratio)

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None