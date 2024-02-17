import random
from game import Game, Move, Player
from minmax import MinMaxPlayer
import sys

def get_dict_size(cache):
        size = sys.getsizeof(cache)/(1024*1024)
        return size

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':
    #g.print()
    player1 = MinMaxPlayer(3, caching=True)
    player2 = RandomPlayer()
    
    while get_dict_size(player1.cache) < 100:
        w1, w2 = 0, 0
        for i in range(100):
            player1.cache_hits = 0
            g = Game()
            winner = g.play(player2, player1)
            if winner == 1:
                w1 += 1
            else:
                w2 += 1
            #print(f'{i}-th game')
        #g.print()
        print(f"Minmaxplayer won {w2/(100)} times.\n CACHE usage: {player1.cache_hits}\nCACHE DIM: {get_dict_size(player1.cache)} MB")
    
    with open('brain.txt', 'w') as f:
        f.write(str(player1.cache))
