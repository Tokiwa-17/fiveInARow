
import numpy as np
import copy
from operator import itemgetter

def rolloutPolicyFn(board):
    """
    :param board:
    :return:
    """
    actionProb = np.random.rand(len(board.legalActions))
    # 返回一个随机分配的概率np.arrays
    return zip(board.legalActions, actionProb)
    # 把move值和概率合成元组列表

def policyValueFn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    actionProbs = np.ones(len(board.legalActions))/len(board.legalActions)
    return zip(board.legalActions, actionProbs), 0

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, priorP):
        self._parent = parent
        self._children = {}  # 下一步棋后会到另外一个TreeNode {move:Treenode}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = priorP

    def expand(self, actionPriors):
        """
        根据一个未完全展开的节点走到一个没有访问过的棋局
        """
        for action, prob in actionPriors:
            if action not in self._children: # 未扩展过
                self._children[action] = TreeNode(self, prob) #新叶子节点的父节点是当前节点，概率是我们预先分配的先验概率

    def select(self, cPuct):
        """
        一个最优可能且没有访问过的局面, actNode 是{move, treenode}类型，返回评估函数中最大的
        """
        return max(self._children.items(),
                   key = lambda actNode: actNode[1].getValue(cPuct))

    def update(self, leafValue):
        """
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leafValue - self._Q) / self._n_visits

    def updateRecursive(self, leafValue):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.updateRecursive(-leafValue)
        self.update(leafValue)

    def getValue(self, cPuct):
        self._u = (cPuct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def isLeaf(self):
        return self._children == {}

    def isRoot(self):
        return self._parent is None


class MCTS(object):


    def __init__(self, policyValueFn, cPuct = 5, nPlayout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policyValueFn
        self._cPuct = cPuct
        self._nPlayout = nPlayout

    def _playout(self, state):

        node = self._root
        while(1):
            if node.isLeaf():

                break

            action, node = node.select(self._cPuct) # move, treeNode
            state.doMove(action)

        actionProbs, _ = self._policy(state)


        end, winner = state.gameEnd()
        if not end:
            node.expand(actionProbs)
        leafValue = self._evaluateRollout(state)
        node.updateRecursive(-leafValue)

    def _evaluateRollout(self, state, limit = 1000):
        """
        Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.getCurrentPlayer()
        for i in range(limit):
            end, winner = state.gameEnd()
            if end:
                break
            actionProbs = rolloutPolicyFn(state) # actionProb (move, prob)的元组列表
            maxAction = max(actionProbs, key = itemgetter(1))[0]
            state.doMove(maxAction)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def getMove(self, state):
        """
        Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._nPlayout):
            # 统计访问最多的move
            stateCopy = copy.deepcopy(state)
            self._playout(stateCopy)
        return max(self._root._children.items(),
                   key=lambda actNode: actNode[1]._n_visits)[0]

    def updateWithMove(self, lastMove):
        if lastMove in self._root._children:
            self._root = self._root._children[lastMove]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, cPuct = 5, nPlayout=2000):
        self.mcts = MCTS(policyValueFn, cPuct, nPlayout)

    def setPlayerInd(self, p):
        self.player = p

    def resetPlayer(self):
        self.mcts.updateWithMove(-1)

    def getAction(self, board):
        sensibleMoves = board.legalActions
        if len(sensibleMoves) > 0:
            move = self.mcts.getMove(board)
            self.mcts.updateWithMove(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
