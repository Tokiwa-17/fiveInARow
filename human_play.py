from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from police_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet

class Human(object):
    def __init__(self):
        self.player = None

    def setPlayerInd(self, p):
        self.player = p

def run():
    n, width, height = 5, 8, 8
    model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width = width, height = height, n_in_row = n)
        game = Game(board)
        policy_param = pickle.load(open(model_file, 'rb'), encoding = 'bytes')
        bestPolicy = PolicyValueNetNumpy(width, height, policy_param)
        mctsPlayer = MCTSPlayer(bestPolicy.policyValueFn, cPuct = 5, nPlayout = 400)
        human = Human()
        game.startPlay(human, mctsPlayer, startPlayer = 1, isShown = 1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()