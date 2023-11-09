from functools import partial
import logging
from pprint import pprint, pformat
from collections import namedtuple
import random
from copy import deepcopy
from typing import Callable, Tuple
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

Genotype = namedtuple("Nimply", "row, num_objects")
Population = list[Genotype]
FitnessFunction = Callable[[Genotype], float]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunction], Tuple[Genotype, Genotype]]
MutationFunc = Callable[[Genotype], Genotype]

def generate_genotype(nrows: int ) -> Genotype:
    """A genotype is represented as a tuple (row, action), so generate a random genome respecting game constraints"""
    row = random.randint(0, nrows - 1)
    action = random.randint(1, row * 2 + 1)
    return Nimply(row, action)

def generate_population(size: int, nrows: int) -> Population:
    """Population is a list of genotypes"""
    return [generate_genotype(nrows) for _ in range(size)]

def fitness(genotype: Genotype, state: Nim):
    """Scores the "goodness" of a genotype if it leads to a nim sum"""

    post_genotype_state = list(state.rows)
    post_genotype_state[genotype.row] -= genotype.num_objects
    #Check if invalid move, so returns -1 -> discard the move
    if (any(n < 0 for n in post_genotype_state)):
        return -1
    
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in post_genotype_state])
    xor = tmp.sum(axis=0) % 2
    xor = int("".join(str(_) for _ in xor), base=2)

    return xor

def selection_pair(population: Population, fitness_fun: FitnessFunction) -> Population:
    return random.choices(
        population=population,
        weights=[fitness_fun(genotype) for genotype in population],
        k=10
    )


def mutation(genotype: Genotype, num = 1,  probability = 0.5):
    for _ in range(num):
        num_objects = genotype.num_objects if random.random() > probability else abs(genotype.num_objects - 1)
    return Nimply(genotype.row, num_objects)

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunction,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        mutation_func : MutationFunc = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    
    for i in range(generation_limit):
        print(f"Generation {i}")
        population = sorted(
            population, 
            key=lambda genotype: fitness_func(genotype),
            reverse = False
        )

        if fitness_func(population[0]) >= fitness_limit:
            print("limit break")
            break
        
        next_generation = population[0:10]

        for j in range(int(len(population) / 10) - 1):
            parents = selection_func(population, fitness_func)
            for genotype in parents:
                next_generation.append(mutation_func(genotype))
        
        population = next_generation

    population = sorted(
            population, 
            key=lambda genotype: fitness_func(genotype),
            reverse = False
        )
    
    return population, i


nim = Nim(5)

population, generations = run_evolution(
    populate_func = partial(
        generate_population, size = 50, nrows = 5
    ),
    fitness_func = partial(
        fitness, state = nim
    ),
    fitness_limit = 1000
)

print(f"Number of generations: {generations}")
print(f"Solutions: {population}")