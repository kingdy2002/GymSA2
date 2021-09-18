from .Node import MCTsNode
import math
import numpy as np
import copy
import torch

class MCTs(object) :
    def __init__(self, network, c_ratio, n_playout) :
        self.root = MCTsNode(None,1,c_ratio)
        self.network = network
        self.n_playout = n_playout
        self.c_ratio = c_ratio


    def playout(self,board) :
        self.node = self.root
        while True :
            move, self.node = self.node.search()
            board.place(move)
            end, win = board.check_end()

            if end :
                child_q = win
                self.node.update_all(-child_q)
                break
            else :
                with torch.no_grad():
                    states = board.current_state()
                    states = torch.FloatTensor(states).cuda().unsqueeze(0)
                    a_probs, child_q = self.network(states)
                if self.node.is_leaf():
                    self.node.add_chiled(a_probs,board.valid_move)
                    self.node.update_all(-child_q.item())
                    break

    def get_move_probs(self,board):
        for i in range(self.n_playout):
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)
        act_visits = [(act, node.visited_n)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = visits/np.sum(visits)

        return acts, act_probs

    def update_and_restart_mcts_by_move(self,move,board):
        self.root = self.root.children[move]
        self.root.parent = None
        with torch.no_grad():
            states = board.current_state()
            states = torch.FloatTensor(states).cuda().unsqueeze(0)
            a_probs, child_q = self.network(states)
        self.root.add_chiled(a_probs,board.valid_move)


    def reset_mcts(self,board):
        self.root = MCTsNode(None, 1, self.c_ratio)
        with torch.no_grad():
            states = board.current_state()
            states = torch.FloatTensor(states).cuda().unsqueeze(0)
            a_probs, child_q = self.network(states)
        self.root.add_chiled(a_probs,board.valid_move)

    def get_move(self,board,self_play):
        acts, probs = self.get_move_probs(board)
        move_probs = np.zeros(board.width * board.height)
        move_probs[list(acts)] = probs
        if self_play :
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            self.reset_mcts(board)
        else :
            move = np.argmax(move_probs)
            self.reset_mcts(board)

        return move, move_probs