import numpy as np
from random import choice

#Globals
h = 64
threes = [[9, 18, 10, 19, 11, 20],
         [12, 21, 13, 22, 14, 23],
         [15, 24, 16, 25, 17, 26],
         [9, 18, 12, 21, 15, 24],
         [10, 19, 13, 22, 16, 25],
         [11, 20, 14, 23, 17, 26],
         [9, 18, 13, 22, 17, 26],
         [11, 20, 13, 22, 15, 24]]

class Board(object):

    def __init__(self):
        self.state = [int(i < 9) for i in range(0, 27)]
        self.turn = 1
        self.legal_moves = [i for i in range(0, 9)]
        self.history = [[[i for i in self.state], 1, 0]]

    def play(self, move):
        self.state[move] = 0
        self.legal_moves.remove(move)
        if self.turn == 1:
            self.state[move + 9] = 1
        else:
            self.state[move + 18] = -1
        self.turn *= -1
        self.history.append([[i for i in self.state], self.turn, 0])

    def winner(self):
        for i in threes:
            tot = 0
            for j in i:
                tot += self.state[j]
            if abs(tot) == 3:
                return tot / 3
        return 0

    def finish(self, result):
        for i in self.history:
            i[2] = result

class Network(object):

    def __init__(self, **parent):
        if parent:
            pass
        else:
            self.w = 0.01 * np.random.randn(27, h)
            self.b = np.zeros((1, h))
            self.w2 = 0.01 * np.random.randn(h, 9)
            self.b2 = np.zeros((1, 9))

    def choose(self, board):
        inpt = np.asarray(board.state)
        inpt[inpt < 0] = 1
        a = np.maximum(0, np.dot(inpt, self.w) + self.b)
        a2 = np.dot(a, self.w2) + self.b2
        exp_a2 = np.exp(a2)
        exp_a2[board.state[exp_a2] == 1] = 0
        softmax = exp_a2 / np.sum(exp_a2, axis=1, keepdims=True)
        p = softmax.tolist()
        return p.index(max(p))

#Functions
def playout(board, policy_one, policy_two, show):
    while len(board.legal_moves) > 0:
        if board.turn == 1:
            board.play(policy_one(board.legal_moves))
        else:
            board.play(policy_two(board.legal_moves))
        if show == True:
            print(board.state)
        result = board.winner()
        if result != 0:
            if show:
                print(result)
            board.finish(result)
            break


    
        
