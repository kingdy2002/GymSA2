from .Node import Node 
import numpy as np


class Maxheap(object) :
    def __init__(self,buffer_size) :
        self.heap = self.init_heap()
        self.buffer_size = buffer_size

        self.current_size = 0



    def init_heap(self):
        heap = np.array([Node(None, None) for _ in range(self.buffer_size)])
        heap[0] = Node(float("inf"), (None, None, None, None, None))
        return heap

    def exch(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]


    def update_node(self,index,key,value) :
        self.deque[index].update_key(key)
        self.deque[index].update_key(value)

    def less(tar, sou) :
        return self.heap[tar].key < self.heap[sou].key 

    def swim(self,index) :
        while index > 1 and less(index //2 ,index) :
            exch(index,index//2)
            index = index//2


    def sink(self, index) :
        while index < self.current_size :
            child = index *2 
            if child >= self.current_size :
                break
            if less(child, child+1) :
                child += 1
            if less(child,index) :
                break
        
            exch(child,index)
            index = child

    def input(self, key,value) :
        if self.current_size == self.buffer_size :
            self.update_node(self.current_size-1,key,value)
            self.swim(self.current_size-1)

        else :
            self.update_node(self.current_size,key,value)
            self.swim(self.current_size)
            self.current_size += 1
