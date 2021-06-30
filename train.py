from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

class TrainPipeline():
    def __init__(self, initModel = None):
        # params of the board and the game
        self.boardWidth = 8
        self.boardHeight = 8
        self.nInRow = 5
        self.board = Board(width = self.boardWidth, height = self.boardHeight, n_in_row = self.nInRow)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0

        self.pure_mcts_playout_num = 1000
        if initModel:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.boardWidth, self.boardHeight, modelFile = initModel)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.boardWidth, self.boardHeight)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policyValueFn,
                                      cPuct = self.c_puct, nPlayout = self.n_playout, isSelfplay = 1)

    def get_equi_data(self, play_data):

        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.boardHeight, self.boardWidth)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))

                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games = 1):
        for i in range(n_games):
            winner, play_data = self.game.startSelfPlay(self.mcts_player,
                                                          temp = self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policyValue(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.trainStep(state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policyValue(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis = 1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policyValueFn,
                                         cPuct=self.c_puct,
                                         nPlayout=self.n_playout)
        pure_mcts_player = MCTS_Pure(cPuct = 5,
                                     nPlayout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.startPlay(current_mcts_player,
                                          pure_mcts_player,
                                          startPlayer = i % 2,
                                          isShown = 0)
            win_cnt[winner] += 1 # win_cnt 字典类型
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    # print(">")
                    self.policy_update()
                    loss, entropy = self.policy_update()
                    if i >= 0 and (i + 1) % 10 == 0:
                        print("batch i:{}, loss {}, entropy {}".format(i + 1, loss, entropy))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.saveModel('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.saveModel('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    trainingPipeline = TrainPipeline()
    trainingPipeline.run()