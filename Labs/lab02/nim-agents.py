from functools import partial
import logging
from pprint import pprint, pformat
from collections import namedtuple
import random
from copy import deepcopy, copy
from typing import Callable, List, Set, Tuple
import numpy as np


Nimply = namedtuple("Nimply", "row, num_objects")

class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    #to play a move
    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


"""
For simplicity, the game is fixed to 5 rows
"""
POPULATION_SIZE = 73_000
GENERATIONS = 100
TOURNAMENT_SIZE = 10
OFFSPRING_SIZE = 100
MUTATION_PROBABILITY = 0.20

class Individual():
    def __init__(self, s=None, m=None, f=None):
        if(s):
            self.states = s
            self.moves = m
            self.scores = f
        else:
            self.states = generate_state_list()
            self.moves = []
            self.scores = []
            self.create_moveset()
            self.init_fitness()  
        self.fit = None
        self.score()
        if(s):
            print(f'\n Individual generated with scores: {self.scores} and fitness {self.fit}')
        #print("scores", self.scores, "fit", self.fit)

    def __str__(self):
        return f"Individual with total fitness of {self.fit}:\nstates: {self.states},\nmoves: {self.moves},\nscores: {self.scores}"
    
    def __repr__(self):
        return f"\nIndividual with total fitness of {self.fit}:\nstates: {self.states},\nmoves: {self.moves},\nscores: {self.scores}"

    def create_moveset(self):
        for j, state in enumerate(self.states):
            rows = []
            for i, row in enumerate(state):
                if row != 0:
                    rows.append(i)
            row = random.choice(rows)
            num_objects = random.randint(1, state[row])
            self.moves.append(Nimply(row, num_objects))

    def init_fitness(self):
        for i, state in enumerate(self.states):
            current = list(state)
            current[self.moves[i].row] -= self.moves[i].num_objects 
            tmp = np.array([tuple(int(x) for x in f"{c:04b}") for c in current])
            xor = tmp.sum(axis=0) % 2
            xor = int("".join(str(_) for _ in xor), base=2)
            self.scores.append(xor)
            
    def fitness(self, idx):
        current = list(self.states[idx])
        current[self.moves[idx].row] -= self.moves[idx].num_objects
        tmp = np.array([tuple(int(x) for x in f"{c:04b}") for c in current])
        xor = tmp.sum(axis=0) % 2
        xor = int("".join(str(_) for _ in xor), base=2)
        self.scores[idx] = xor

    def score(self):
        self.fit = sum(self.scores)
    

Population = List[Individual]

def generate_move(state):
    rows = []
    for i, row in enumerate(state):
        if row != 0:
            rows.append(i)
    row = random.choice(rows)
    num_objects = random.randint(1, state[row])
    move =  Nimply(row, num_objects)
    return move

def generate_state(prev = None):
    if(prev):
        state = list(prev)
        rows = []
        for i, row in enumerate(state):
            if row != 0:
                rows.append(i)
        row = random.choice(rows)
        num_objects = random.randint(1, state[row])
        state[row] -= num_objects

    else:
        state = [1,3,5,7,9]
        if random.random() > 0.5:
            i = random.randint(0, 4)
            state[i] = random.randint(1, i*2+1)
    return tuple(state)

def generate_state_list():
    states = []
    states.append(generate_state())
    while not (sum(1 for o in list(states[-1]) if o > 0) == 1):
        states.append(generate_state(states[-1]))
    return states

def generate_population() -> Population:
    return [Individual() for _ in range(POPULATION_SIZE)]

def select_parent(population):
    pool = [random.choice(population) for _ in range(TOURNAMENT_SIZE)]
    champion = min(pool, key=lambda i: i.fit)
    return champion

def mutation(ind: Individual) -> Individual:
    offspring = copy(ind)
    idx = random.randint(0, len(offspring.states) - 1)
    offspring.moves[idx] = generate_move(offspring.states[idx])
    offspring.fitness(idx)
    return offspring

def crossover(inds: List[Individual]):
    """Fixed 2 inds"""
    p1 = inds[0]
    p2 = inds[1]
 
    s = p1.states[0:(len(p1.states)//2)] + p2.states[(len(p2.states)//2):]
    m = p1.moves[0:(len(p1.states)//2)] + p2.moves[(len(p2.states)//2):]
    f = p1.scores[0:(len(p1.states)//2)] + p2.scores[(len(p2.states)//2):]
    #print(s,m,f)

    return Individual(s, m, f)

"""
population = generate_population()

for generation in range(GENERATIONS):
    offsprings = []
    for _ in range(OFFSPRING_SIZE):
        if random.random() < MUTATION_PROBABILITY:
            p = select_parent(population)
            o = mutation(p)
        else:
            o = crossover([select_parent(population) for _ in range(2)])
        offsprings.append(o)
    population.extend(offsprings)
    population.sort(key=lambda i: i.fit, reverse=False)
    population = population[:POPULATION_SIZE]
   
    best = min(population, key=lambda o: o.fit)
    print()
    print(f"Generation {generation}, minimum fitness offspring of this gen: {best}")
"""

class Agent():
    def __init__(self):
        self.brain = None
    
    def training(self):
        population = generate_population()

        for generation in range(GENERATIONS):
            offsprings = []
            for _ in range(OFFSPRING_SIZE):
                if random.random() < MUTATION_PROBABILITY:
                    p = select_parent(population)
                    o = mutation(p)
                else:
                    o = crossover([select_parent(population) for _ in range(2)])
                offsprings.append(o)
            population.extend(offsprings)
            population.sort(key=lambda i: i.fit, reverse=False)
            population = population[:POPULATION_SIZE]
        
            best = min(population, key=lambda o: o.fit)
            print()
            print(f"Generation {generation}, minimum fitness offspring of this gen: {best}")

        self.brain = population

    def play(self, nim: Nim):
        move = None
        playable_moves = set()
        fitness = 999
        for neuron in self.brain:
            for i, state in enumerate(neuron.states):
                if state == nim:
                    playable_moves.add(tuple([neuron.moves[i], neuron.scores[i]]))
                    if neuron.scores[i] < fitness:
                        move = neuron.moves[i]
                        fitness = neuron.scores[i]
        print(f'Available moves: {playable_moves}')
        if move:
            return move
        else:
            return generate_move(nim)

ciccio = Agent()
ciccio.training()

nim = Nim(5)
player = random.randint(0,1)
print(f"init : {nim}")
while nim:
    player = 1 - player
    if player:
        pile = int(input("Choose Pile: "))
        count = int(input("Choose Count: "))
        ply = Nimply(pile, count)
        print(f"You chose to take {ply.num_objects} from pile {ply.row}.")
    else:
        ply = ciccio.play(nim.rows)
        print(f"AI chose to take {ply.num_objects} from pile {ply.row}.")

    nim.nimming(ply)
    print(f"status: {nim}")
if(player):    
    print(f"status: You won!")
else:
    print(f"status: AI won!")