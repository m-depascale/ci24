from collections import namedtuple
from copy import copy
from pprint import pprint
import time
import numpy as np
import random 

Ply = namedtuple("ToeMove", "i, j")

class TicTacToe:
    def __init__(self):
        self.hidden_board = np.array([6,7,2, 1,5,9, 8,3,4])
        self.board = np.zeros((1,9), dtype=np.int32)
        self.player = 1
        self.isEnd = False
        self.winner = None

    def isAvailable(self, move):
        return True if self.board[0][move] == 0 else False
    
    @staticmethod
    def available_actions(board):
        moves = []
        for idx, el in enumerate(board.tolist()[0]): 
            if el == 0:
                moves.append(idx) 
        return moves

    def print_tic_tac_toe_board(self):
        print("-----------")
        for row in np.reshape(self.board, (3,3)):
            print("|".join([' X ' if val > 0 else ' O ' if val < 0 else ' . ' for val in row]))
            print("-----------")

    def switch_player(self):
        self.player = -1 * self.player

    def check_end(self):
        board = np.reshape(self.board, (3,3))
        for i in range(3):
            if np.sum(board[:, i]) == 15 or np.sum(board[:, i]) == -15: #column sum
                #print("WIN")
                self.winner = self.player
                self.isEnd = True
                return
            if np.sum(board[i]) == 15 or np.sum(board[i]) == -15: #raw sum
                #print("WIN")
                self.winner = self.player
                self.isEnd = True
                return
            if np.trace(board[:, ::-1]) == 15 or np.trace(board[:, ::-1]) == -15:
                #print("WIN")
                self.winner = self.player
                self.isEnd = True
                return
            if np.trace(board) == 15 or np.trace(board) == -15:
                #print("WIN")
                self.winner = self.player
                self.isEnd = True
                return
        if not np.any(self.board == 0): 
            self.winner = None 
            self.isEnd = True

    def move(self, move):
        if self.isAvailable(move):
            self.board[0][move] = self.player * self.hidden_board[move]
            self.check_end()
            #print(self.winner)
            if not self.isEnd:
                self.switch_player()


class TicTacToeAgent():
    #model free q learning with some minmax strategy
    def __init__(self, alpha=0.5, epsilon=0.3, discount_factor=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_Q_value(self, board, action):
        key = (tuple(board), action)
        return self.q.get(key, 0.0)

    def minmax(self, node, depth, player):
        if depth == 0 or node.isTerminal: 
            return 0
        if player == 1: #max
            return 
        return 0

    def choose_action(self, state, epsilon=False):
        # state = board
        available_actions = TicTacToe.available_actions(state.board)
        if epsilon and random.uniform(0, 1) < self.epsilon:
            action = random.choice(available_actions)
        else:
            Q_values = [self.get_Q_value(state.board.tolist()[0], a) for a in available_actions]
            action = available_actions[Q_values.index(max(Q_values))]
        return action
    
    def reward(self, state):
        return state.winner
    
    def update_Q_value(self, state, action, reward, next_state):
        if reward is None:
            reward = -0.1 
        #print(reward)
        state_key = tuple(state.board.tolist()[0])
        next_state_key = tuple(next_state.board.tolist()[0])
        next_Q_values = [self.get_Q_value(next_state_key, next_action) for next_action in TicTacToe.available_actions(next_state.board)]
        max_next_Q = max(next_Q_values) if next_Q_values else 0.0
        self.q[(state_key, action)] = self.q.get((state_key, action), 0.0) + self.alpha * (reward + self.discount_factor * max_next_Q - self.q.get((state_key, action), 0.0))



def train(n=1000):
    """
    Train an AI by playing `n` games against itself.
    """
    ai = TicTacToeAgent()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = TicTacToe()

        # Game loop
        while not game.isEnd:
            #game.print_tic_tac_toe_board()
            
            # Keep track of current state and action
            state = copy(game)
            action = ai.choose_action(state, epsilon=True)

            # Make move
            game.move(action)
            new_state = copy(game)

            # When game is over, update Q values with rewards
            ai.update_Q_value(state, action, new_state.winner, new_state)

    print("Done training")

    # Return the trained AI
    return ai


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """
    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = TicTacToe()

    # Game loop
    while True:
        # Print contents of piles
        print()
        print("Board:")
        game.print_tic_tac_toe_board()
        print()

        # Compute available actions
        available_actions = TicTacToe.available_actions(game.board)
        time.sleep(1)
        player = 1 if game.player == 1 else 0
        # Let human make a move
        if player == human_player:
            print("Your Turn")
            while True:
                i = int(input("Choose move (from 0 to 8): "))
                if i in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            i = ai.choose_action(game, epsilon=False)
            print(f"AI chose to write in position {i}")

        # Make move
        game.move(i)

        # Check for winner
        if game.isEnd:
            print()
            print("GAME OVER")
            winner = "Human" if player == human_player else "AI"
            print(f"Winner is {winner}")
            return
        
ciccio =train(10000)
pprint(ciccio.q)
play(ciccio)
play(ciccio)