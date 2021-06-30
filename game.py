from __future__ import print_function
import numpy as np
import tkinter

class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))

        self.states = {}
        '''{key:value} 
        key: move y + x * width
        value: 1 or 2        
        '''

        self.n_in_row = int(kwargs.get('n_in_row', 5))  # num of pieces for win
        self.players = [1, 2]  # player1 and player2

    def initBoard(self, startPlayer = 0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        # board is too small

        self.currentPlayer = self.players[startPlayer]  # start player
        # keep available moves in a list
        self.legalActions = list(range(self.width * self.height))
        self.states = {}
        self.lastMove = -1

    def moveToLocation(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        # from left-down to right-up
        h = move // self.width
        w = move % self.width
        return [h, w]

    def locationToMove(self, location):
        """
        :param location:
        :return: reverse function of moveToLocation
        """
        if len(location) != 2: # length of list < 2
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def currentState(self):
        """
        :return: the board state from the perspective of the current player. 4*width*height
        """
        squareState = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # 2lists: moves = location, players
            # classify the location to 2 sorts
            moveCurr = moves[players == self.currentPlayer]
            moveOppo = moves[players != self.currentPlayer]

            squareState[0][moveCurr // self.width,
                            moveCurr % self.height] = 1.0
            # 把当前玩家的棋子 (x / width, y % height)
            squareState[1][moveOppo // self.width,
                            moveOppo % self.height] = 1.0
            # indicate the last move location
            squareState[2][self.lastMove // self.width,
                            self.lastMove % self.height] = 1.0
        if len(self.states) % 2 == 0:
            squareState[3][:, :] = 1.0  # indicate the colour to play
        return squareState[:, ::-1, :] # ::-1 -1是索引步长，从上向下输出棋盘

    def doMove(self, move):
        """
        simulates the process of game
        :param move:
        :return:
        """
        self.states[move] = self.currentPlayer
        self.legalActions.remove(move)
        self.currentPlayer = (
            self.players[0] if self.currentPlayer == self.players[1]
            else self.players[1]
        )
        self.lastMove = move

    def isWin(self):
        """
        :return: (whether win, id of player)
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.legalActions))
        # 已经下过棋的位置列表

        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # 横
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1): # (m, m + n)棋子的类型数为1
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def gameEnd(self):
        """Check whether the game is ended or not"""
        win, winner = self.isWin()
        if win:
            return True, winner
        elif not len(self.legalActions):
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        return self.currentPlayer

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pixel_x = 30 + 30 * self.x # 中心点坐标， 一个格子三十像素
        self.pixel_y = 30 + 30 * self.y

class Game(object):
    """
    game server
    """

    def __init__(self, board, **kwargs):
        self.board = board

    def click1(self, event):

        currentPlayer = self.board.getCurrentPlayer()
        if currentPlayer == 1:
            # 点击的格子
            i = (event.x) // 30
            j = (event.y) // 30
            # 判断离哪个点更近
            ri = (event.x) % 30
            rj = (event.y) % 30
            i = i - 1 if ri < 15 else i
            j = j - 1 if rj < 15 else j

            move = self.board.locationToMove((i, j))
            if move in self.board.legalActions:
                self.cv.create_oval(self.chess_board_points[i][j].pixel_x - 10,
                                    self.chess_board_points[i][j].pixel_y - 10,
                                    self.chess_board_points[i][j].pixel_x + 10,
                                    self.chess_board_points[i][j].pixel_y + 10, fill='black')
                self.board.doMove(move)

    """TODO"""
    def run(self):
        currentPlayer = self.board.getCurrentPlayer()

        end, winner = self.board.gameEnd()

        if currentPlayer == 2 and not end:
            player = self.players[currentPlayer]
            move = player.getAction(self.board) # getAction 应该在别的类 player 类型？
            self.board.doMove(move)
            i, j = self.board.moveToLocation(move)
            self.cv.create_oval(self.chess_board_points[i][j].pixel_x - 10, self.chess_board_points[i][j].pixel_y - 10,
                                self.chess_board_points[i][j].pixel_x + 10, self.chess_board_points[i][j].pixel_y + 10,
                                fill='white')

        end, winner = self.board.gameEnd()

        if end:
            if winner != -1: # 非平局
                self.cv.create_text(self.board.width * 15 + 15, self.board.height * 30 + 30,
                                    text="Game over. Winner is {}".format(self.players[winner]))
                self.cv.unbind('<Button-1>')
            else:
                self.cv.create_text(self.board.width * 15 + 15, self.board.height * 30 + 30, text="Game end. Tie")

            return winner
        else:
            self.cv.after(100, self.run)

    def graphic(self, board, player1, player2):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height

        p1, p2 = self.board.players
        player1.setPlayerInd(p1)
        player2.setPlayerInd(p2)
        self.players = {p1: player1, p2: player2}

        window = tkinter.Tk()
        self.cv = tkinter.Canvas(window, height=height * 30 + 60, width=width * 30 + 30, bg='white')
        self.chess_board_points = [[None for i in range(height)] for j in range(width)]

        for i in range(width):
            for j in range(height):
                self.chess_board_points[i][j] = Point(i, j);
        for i in range(width):  # vertical line
            self.cv.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y,
                                self.chess_board_points[i][width - 1].pixel_x,
                                self.chess_board_points[i][width - 1].pixel_y)

        for j in range(height):  # rizontal line
            self.cv.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y,
                                self.chess_board_points[height - 1][j].pixel_x,
                                self.chess_board_points[height - 1][j].pixel_y)

        self.button = tkinter.Button(window, text="start game!", command=self.run)
        self.cv.bind('<Button-1>', self.click1)
        self.cv.pack()
        self.button.pack()
        window.mainloop()

    def startPlay(self, player1, player2, startPlayer = 0, isShown = 1):
        """
        start a game between two players
        """
        if startPlayer not in (0, 1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.initBoard(startPlayer)

        if isShown:
            self.graphic(self.board, player1, player2)
        else:
            p1, p2 = self.board.players
            player1.setPlayerInd(p1)
            player2.setPlayerInd(p2)
            players = {p1: player1, p2: player2}
            while (1):
                currentPlayer = self.board.getCurrentPlayer()
                print(currentPlayer)
                player = players[currentPlayer]
                move = player.getAction(self.board)
                self.board.doMove(move)
                if isShown:
                    self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.gameEnd()
                if end:
                    return winner

    def startSelfPlay(self, player, isShown = 0, temp = 1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        self.board.initBoard()
        p1, p2 = self.board.players
        states, mctsProbs, currentPlayers = [], [], []
        while (1):
            move, moveProbs = player.getAction(self.board, temp = temp, returnProb = 1)
            # store the data
            states.append(self.board.currentState())
            mctsProbs.append(moveProbs)
            currentPlayers.append(self.board.currentPlayer)
            # perform a move
            self.board.doMove(move)
            end, winner = self.board.gameEnd()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(currentPlayers))
                if winner != -1:
                    winners_z[np.array(currentPlayers) == winner] = 1.0
                    winners_z[np.array(currentPlayers) != winner] = -1.0
                # reset MCTS root node
                player.resetPlayer()
                if isShown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mctsProbs, winners_z)