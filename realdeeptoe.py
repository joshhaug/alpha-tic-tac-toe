#Imports

import numpy as np
from random import choice, sample

#Hello Josh

#Globals

h = 64 #Number of hidden neurons
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
        # state is of the form [played, player_1_moves, player_2_moves]
        # each of the above sections is 9 elements long
        self.state = [int(i < 9) for i in range(0, 27)]
        self.turn = 1
        self.legal_moves = [i for i in range(0, 9)]
        # history is of the form [[[board], turn, winner]]
        self.history = [[self.state[:], 1, 0]]

    # Modifies the game state for some position `move` (between 0 and 9). 
    def play(self, move):
        self.state[move] = 0
        self.legal_moves.remove(move)
        if self.turn == 1:  
            self.state[move + 9] = 1
        else:
            self.state[move + 18] = -1
        self.turn *= -1
        self.history.append([self.state[:], self.turn, 0])
        self.render_josh()

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

    def render(self):
        boxes = ["","","","","","","","",""]
        # fill with player one's moves
        for i in range(9,18):
            boxes[i-9] = "X" if self.state[i] > 0 else " "
        # fill with player two's moves
        for i in range(18,27):
            if self.state[i] < 0:
                boxes[i-18] = "O"
        offset = 0
        # print out board
        print("== BOARD ==")
        for row in range(0,3):
            print boxes[0+offset],'|',boxes[1+offset],'|',boxes[2+offset] 
            print("---------")
            offset += 3
        print("\n")



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
        inpt = np.asarray([board.state])
        inpt[inpt < 0] = 1
        a = np.maximum(0, np.dot(inpt, self.w) + self.b)
        a2 = np.dot(a, self.w2) + self.b2
        exp_a2 = np.exp(a2)
        for i in range(0, 9):
            if inpt[0, i] == 0:
                exp_a2[0, i] = 0
        softmax = exp_a2 / np.sum(exp_a2, axis=1, keepdims=True)
        p = softmax[0].tolist()
        print(p)
        return p.index(max(p))

def random_policy(board):
    return choice(board.legal_moves)

#Functions
def playout(board, policy_one, policy_two, show):
    while len(board.legal_moves) > 0:
        if board.turn == 1:
            board.play(policy_one(board))
        else:
            board.play(policy_two(board))
        if show == True:
            print(board.state)
        result = board.winner()
        if result != 0:
            if show == True:
                print(result)
            board.finish(result)
            break

#Code
network = Network()
playout(Board(), network.choose, network.choose, True)
