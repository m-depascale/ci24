import random
from game import Game, Move, Player
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import sys 

from sim_game import FakeGame

class MinMaxPlayer(Player):
    def __init__(self, depth=3, cache = {}, caching = False) -> None:
        super().__init__()
        self.depth = depth
        self.cache = cache
        self.caching = caching
        self.hits = 0
        self.cache_hits = 0.
        #self.min_depth = min_depth
        #self.max_depth = max_depth

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        fake = FakeGame(game.get_board(), game.get_current_player())
        depth = random.randint(self.depth, 17)
        
        if str(game.get_board()) in self.cache.keys():
            self.hits += 1
            self.cache_hits = self.hits / len(self.cache) 
            if random.random() < 0.5 and self.cache[str(game.get_board())][1] >= depth:
                optimal_move = self.cache[str(game.get_board())][0]
            else:
                _, optimal_move = self.min_max_alpha_beta(fake, -np.infty, np.infty, depth, game.get_current_player())

        else:
            _, optimal_move = self.min_max_alpha_beta(fake, -np.infty, np.infty, depth, game.get_current_player())
        
        if optimal_move is not None:
            from_pos, slide = optimal_move
            if self.caching:
                self.cache[str(game.get_board())] = (optimal_move, depth)
        else:
            from_pos, slide = random.choice(self.available_moves(fake))
        return from_pos, slide


    def min_max_alpha_beta(self, game, alpha, beta, depth, player_idx, max_player = True):
        playing = game.check_winner() ==  -1
        if playing == False or depth == 0:
            winner = game.check_winner()
            p, o = self.board_evaluation(game)
            if winner == player_idx:
                return p - depth, None
            elif winner != player_idx:
                return -o + depth, None
            else:
                return 0, None
        
        optimal = None
        moveset = self.available_moves(game)

        if max_player:
            for move in moveset:
                cooked_board = self.board_after_move(game, move)
                cooked_game = FakeGame(cooked_board, (player_idx + 1) % 2)
                new_value, _ = self.min_max_alpha_beta(cooked_game, alpha, beta, depth-1, (player_idx +1) %2, False)
                if new_value > alpha:
                    alpha = new_value
                    optimal = move
                if alpha >= beta:
                    break
                return alpha, optimal
        else:
            for move in moveset:
                cooked_board = self.board_after_move(game, move)
                cooked_game = FakeGame(cooked_board, (player_idx + 1) % 2)
                new_value, _ = self.min_max_alpha_beta(cooked_game, alpha, beta, depth-1, (player_idx +1) %2, True)
                if new_value < beta:
                    beta = new_value
                    optimal = move
                if alpha >= beta:
                    break
                return beta, optimal

    def available_moves(self, game) -> list(tuple[tuple[int, int], Move]):
        #returns the available moveset
        moveset = []
        for i in range(5):
            for j in range(5):
                for m in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                    ok = game.isMoveValid((i, j), m)
                    if ok:
                        moveset.append(((i, j), m))
        return moveset
    
    def board_after_move(self, game, move):
        new_game = deepcopy(game)
        new_board = new_game.board_after_move(move)
        return new_board
    
    def board_evaluation(self, game):
        ### compute a score for the board
        
        board = game.get_board()
        player = game.get_current_player()
        #scoring: more tiles in a row/col more the score, but also 
        # if there is a tile above-under/left-right a blank or enemy slot: it become skinnyyyyyyyyyyy (very fitness)
        rows_scores = []
        for r in range(board.shape[0]):
            y0 = np.sum(board[r, :] == player)
            z0 = np.sum(board[r, :] == abs(1-player))
            rows_scores.append((y0, z0)) #player, opponent
    
        columns_scores = []
        for c in range(board.shape[1]):
            y1 = np.sum(board[:, c] == player)
            z1 = np.sum(board[:, c] == abs(1-player))
            columns_scores.append((y1,z1)) #player, opponent
        
        diagonals_scores = []
        d0 = np.array([board[x, x] for x in range(board.shape[0])])
        d1 = np.array([board[x, -(x+1)] for x in range(board.shape[0])])
        
        diagonals_scores.append(
            (np.sum(d0 == player), np.sum(d0 == abs(1-player)))
            )
        
        diagonals_scores.append(
            (np.sum(d1 == player), np.sum(d1 == abs(1-player)))
            )
        
        p_score = np.sum([x[0] for x in rows_scores]) + np.sum([x[0] for x in columns_scores]) +  np.sum([x[0] for x in diagonals_scores]) 
        o_score = np.sum([x[1] for x in rows_scores]) + np.sum([x[1] for x in columns_scores]) +  np.sum([x[1] for x in diagonals_scores]) 
        
        return p_score, o_score