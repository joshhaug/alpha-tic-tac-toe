#Imports

import numpy as np
from random import choice, sample

#Hello Josh

#Globals

h = 16 #Number of hidden neurons
eta = 1
pool_limit = 256
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

def render(state):
    boxes = ["","","","","","","","",""]
    # fill with player one's moves
    for i in range(9,18):
        boxes[i-9] = "X" if state[i] > 0 else " "
    # fill with player two's moves
    for i in range(18,27):
        if state[i] < 0:
            boxes[i-18] = "O"
    offset = 0
    # print out board
    print("== BOARD ==")
    for row in range(0,3):
        print(boxes[0+offset],'|',boxes[1+offset],'|',boxes[2+offset]) 
        print("---------")
        offset += 3
    print("\n")



class Network(object):

    def __init__(self, parent):
        if parent == None:
            self.w = 0.01 * np.random.randn(27, h)
            self.b = np.zeros((1, h))
            self.w2 = 0.01 * np.random.randn(h, 9)
            self.b2 = np.zeros((1, 9))
        else:
            self.w = parent.w[:, :]
            self.b = parent.b[:, :]
            self.w2 = parent.w2[:, :]
            self.b2 = parent.b2[:, :]

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
        return p.index(max(p))

    def update(self, batch, ground):
        #This function updates the parameters of the network
        #according to a batch array and a ground array, using
        #the REINFORCE algorithm. The learning rate is the global
        #variable “eta.”

        #Forward pass
        a = np.maximum(0, np.dot(batch, self.w) + self.b)
        a2 = np.dot(a, self.w2) + self.b2
        c = np.exp(a2)

##        for i in range(c.shape[0]):
##            for j in range(9):
##                if batch[i, j] == 0: c[i, j] = 0

        c = c / np.sum(c, axis=0, keepdims=True)

        #Backward pass
        dzeta2 = 1 - c
        dzeta2 /= batch.shape[0]
        dzeta2 = dzeta2 * ground 
        db2 = np.sum(dzeta2, axis=0, keepdims=True)
        dw2 = np.dot(a.T, dzeta2)
        dzeta = np.dot(dzeta2, self.w2.T)
        dzeta[dzeta < 0] = 0
        db = np.sum(dzeta, axis=0, keepdims=True)
        dw = np.dot(batch.T, dzeta)

        #Update
        self.b2 += db2 * eta
        self.w2 += dw2 * eta
        self.b += db * eta
        self.w += dw * eta

#Functions

def random_policy(board):
    return choice(board.legal_moves)

def playout(board, policy_one, policy_two, show):
    while len(board.legal_moves) > 0:
        if board.turn == 1:
            board.play(policy_one(board))
        else:
            board.play(policy_two(board))
        if show == True:
            render(board.state)
        result = board.winner()
        if result != 0:
            if show == True:
                print(result)
            board.finish(result)
            break

def epoch(policy, opp_policies):
    #This function plays a policy against each policy
    #in a list of opponent policies twice, once as
    #first player and once as second player. It returns
    #all the states with the policy to play from each
    #non-drawn game, in a single n by 27 array, where n
    #is the number of states. It also returns the “ground”
    #which is a n by 1 array (each element is either 1 or -1,
    #corresponding to a win or loss for the policy from that state).

    batch = []
    ground = []
    for i in opp_policies:
        board = Board()
        playout(board, policy, i, False) # first player
        for j in board.history:
            if j[1] == 1:
                batch.append(j[0]) 
                ground.append(j[1] * j[2])
        board = Board()
        playout(board, i, policy, False) # second player
        for j in board.history:
            if j[1] == -1:
                batch.append(j[0])
                ground.append(j[1] * j[2])
    batch = np.asarray(batch)
    ground = np.array([[i] for i in ground])
    return batch, ground

def train(network, num_batches, batch_size):
    #This function trains a neural network using the REINFORCE algorithm
    #for n batches, where n = num_batches, each of which typically involves
    #the playing out of s times 2 games, where s is the batch_size. A pool
    #of opponents contains older versions of the network with a maximum size
    #equal to a global variable, `pool_limit`.

    old_networks = []
    for i in range(num_batches):
        if len(old_networks) == 0: # playing yourself
            batch, ground = epoch(network.choose, [network.choose])
        elif len(old_networks) < batch_size: # haven’t met it yet
            batch, ground = epoch(network.choose, [j.choose for j in
                                                   old_networks])
        else: # take a random subset of the old ones 
            batch, ground = epoch(network.choose, [j.choose for j in
                                                   sample(old_networks,
                                                          batch_size)])
        if len(old_networks) >= pool_limit:
            del old_networks[0]
        old_networks.append(Network(network))
        network.update(batch, ground)
        board = Board()
        playout(board, network.choose, network.choose, True)

def test_learning(network):
    batch, ground = epoch(network.choose, [network.choose])
    network.update(batch, ground)
    print('Updated')
    for i in range(batch.shape[0]):
        print(batch[i, :])
        inpt = np.asarray([batch[i, :]])
        inpt[inpt < 0] = 1
        a = np.maximum(0, np.dot(inpt, network.w) + network.b)
        a2 = np.dot(a, network.w2) + network.b2
        exp_a2 = np.exp(a2)
        for i in range(0, 9):
            if inpt[0, i] == 0:
                exp_a2[0, i] = 0
        softmax = exp_a2 / np.sum(exp_a2, axis=1, keepdims=True)
        p = softmax[0].tolist()
        print(p)
        
