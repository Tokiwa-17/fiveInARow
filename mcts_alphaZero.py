
import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, priorP):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._nVisits = 0
        self._Q = 0
        self._u = 0
        self._P = priorP

    def expand(self, actionPriors):
        """
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in actionPriors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, cPuct):
        """
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda actNode: actNode[1].getValue(cPuct))

    def update(self, leafValue):
        """
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._nVisits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leafValue - self._Q) / self._nVisits

    def updateRecursive(self, leafValue):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.updateRecursive(-leafValue)
        self.update(leafValue)

    def getValue(self, cPuct):
        self._u = (cPuct * self._P *
                   np.sqrt(self._parent._nVisits) / (1 + self._nVisits))
        return self._Q + self._u

    def isLeaf(self):
        return self._children == {}

    def is_Root(self):
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
            # Greedily select next move.
            action, node = node.select(self._cPuct)
            state.doMove(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        actionProbs, leafValue = self._policy(state)
        # Check for end of game.
        end, winner = state.gameEnd()
        if not end:
            node.expand(actionProbs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leafValue = 0.0
            else:
                leafValue = (
                    1.0 if winner == state.getCurrentPlayer() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.updateRecursive(-leafValue)

    def getMoveProbs(self, state, temp = 1e-3):
        """
        :param state:
        :param temp:
        :return:moves,probs
        """
        # 概率计算出来，不再是随便分配的
        for n in range(self._nPlayout):
            stateCopy = copy.deepcopy(state)
            self._playout(stateCopy)

        # calc the move probabilities based on visit counts at the root node
        actVisits = [(act, node._nVisits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*actVisits) # move, 访问次数
        actProbs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, actProbs

    def updateWithMove(self, lastMove):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if lastMove in self._root._children:
            self._root = self._root._children[lastMove]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policyValueFunction,
                 cPuct = 5, nPlayout = 2000, isSelfplay = 0):
        self.mcts = MCTS(policyValueFunction, cPuct, nPlayout)
        self._isSelfplay = isSelfplay

    def setPlayerInd(self, p):
        self.player = p

    def resetPlayer(self):
        self.mcts.updateWithMove(-1)

    def getAction(self, board, temp = 1e-3, returnProb = 0):
        sensibleMoves = board.legalActions
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        moveProbs = np.zeros(board.width*board.height)
        if len(sensibleMoves) > 0:
            acts, probs = self.mcts.getMoveProbs(board, temp)
            moveProbs[list(acts)] = probs # 根据move作为下标作为概率
            if self._isSelfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.updateWithMove(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p = probs)
                # reset the root node
                self.mcts.updateWithMove(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if returnProb:
                return move, moveProbs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)