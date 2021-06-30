import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def setLearningRate(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

class Net(nn.Module):
    def __init__(self, boardWidth, boardHeight):
        super(Net, self).__init__()

        self.boardWidth = boardWidth
        self.boardHeight = boardHeight
        self.conv1 = nn.Conv2d(4, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        # action policy layers
        self.actConv1 = nn.Conv2d(128, 4, 1)
        self.actFc1 =nn.Linear(4 * boardWidth * boardHeight, 64)
        # state value layers
        self.valConv1 = nn.Conv2d(128, 2, 1)
        self.valFc1 = nn.Linear(2 * boardWidth * boardHeight, 64)
        self.valFc2 = nn.Linear(64, 1)

    def forward(self, stateInput):
        x = F.relu(self.conv1(stateInput))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        xAct = F.relu(self.actConv1(x))
        xAct = xAct.view(-1, 4 * self.boardWidth * self.boardHeight)
        xAct = F.log_softmax(self.actFc1(xAct))
        # state value layers
        xVal = F.relu(self.valConv1(x))
        xVal = xVal.view(-1, 2 * self.boardWidth * self.boardHeight)
        xVal = F.relu(self.valFc1(xVal))
        xVal = torch.tanh(self.valFc2(xVal))
        return xAct, xVal

class PolicyValueNet():
    def __init__(self, boardWidth, boardHeight, modelFile = None, useGpu = True):
        self.boardWidth = boardWidth
        self.boardHeight = boardHeight
        self.usegpu = useGpu
        self.l2Const = 1e-4

        if self.usegpu:
            self.policyValueNet = Net(boardWidth, boardHeight).cuda()
        else:
            self.policyValueNet = Net(boardWidth, boardHeight)

        self.optimizer = optim.Adam(self.policyValueNet.parameters(), weight_decay = self.l2Const)

        if modelFile:
            Params = torch.load(modelFile)
            self.policyValueNet.load_state_dict(Params)

    def policyValue(self, stateBatch):
        if self.usegpu:
            stateBatch = Variable(torch.FloatTensor(stateBatch).cuda())
            logActProbs, value = self.policyValueNet(stateBatch)
            actProbs = np.exp(logActProbs.data.cpu().numpy())
            return actProbs, value.data.cpu().numpy()
        else:
            stateBatch = Variable(torch.FloatTensor(stateBatch))
            logActProbs, value = self.policyValueNet(stateBatch)
            actProbs = np.exp(logActProbs.data.cpu().numpy())
            return actProbs, value.data.numpy()

    def policyValueFn(self, board):
        """
        :param board:
        :return: a list of (action, probability) tuples for each available action and the score
        """
        legalPositions = board.legalActions
        currentState = np.ascontiguousarray(board.currentState().reshape(
            -1, 4, self.boardWidth, self.boardHeight))
        if self.usegpu:
            logActProbs, value = self.policyValueNet(
                Variable(torch.from_numpy(currentState)).cuda().float())
            actProbs = np.exp(logActProbs.data.cpu().numpy().flatten())
        else:
            logActProbs, value = self.policyValueNet(
                Variable(torch.from_numpy(currentState)).float())
            actProbs = np.exp(logActProbs.data.numpy().flatten())
        actProbs = zip(legalPositions, actProbs[legalPositions])
        value = value.data[0][0]
        return actProbs, value

    def trainStep(self, stateBatch, mctsProbs, winBatch, lr):
        if self.usegpu:
            stateBatch = Variable(torch.FloatTensor(stateBatch).cuda())
            mctsProbs = Variable(torch.FloatTensor(mctsProbs).cuda())
            winBatch = Variable(torch.FloatTensor(winBatch).cuda())
        else:
            stateBatch = Variable(torch.FloatTensor(stateBatch))
            mctsProbs = Variable(torch.FloatTensor(mctsProbs))
            winBatch = Variable(torch.FloatTensor(winBatch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        setLearningRate(self.optimizer, lr)
        # forward
        logActProbs, value = self.policyValueNet(stateBatch)
        #loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        valueLoss = F.mse_loss(value.view(-1), winBatch)
        policyLoss = -torch.mean(torch.sum(mctsProbs * logActProbs, 1))
        loss = valueLoss + policyLoss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(
                torch.sum(torch.exp(logActProbs) * logActProbs, 1)
                )
        return loss.item(), entropy.item()

    def getPolicyParam(self):
        netParams = self.policyValueNet.state_dict()
        return netParams

    def saveModel(self, modelFile):
        netParams = self.getPolicyParam()  # get model params
        torch.save(netParams, modelFile)