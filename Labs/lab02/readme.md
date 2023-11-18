# Nim AI Training Project

This project is focused on training an AI agent to play the game of Nim using genetic algorithms.

## Overview

The project includes an AI agent that uses genetic algorithms to evolve and improve its strategy over multiple generations. The Nim game is played against a basic AI opponent or a human player.

## Features

- Genetic algorithm for evolving strategies, with plus strategy.
- Nim game implementation.
- Training process to improve the AI's performance.

## Classes and Functions

Note that some of the parameters are chosen just for speed-up the process while debugging. 
### Classes

1. **Nim:**
   - Description: Represents the Nim game.
   - Methods:
     - `__init__(self, num_rows: int, k: int = None) -> None`
     - `__bool__(self) -> bool`
     - `__str__(self) -> str`
     - `rows(self) -> tuple`
     - `nimming(self, ply: Nimply) -> None`

2. **Individual:**
   - Description: Represents an individual in the genetic algorithm population.
   - Methods:
     - `__init__(self, s=None, m=None, f=None) -> None`
     - `__str__(self) -> str`
     - `__repr__(self) -> str`
     - `create_moveset(self) -> None`
     - `init_fitness(self) -> None`
     - `fitness(self, idx) -> None`
     - `score(self) -> None`

3. **Agent:**
   - Description: Represents the AI agent using genetic algorithms, in this case the population is randomly generated and then optimized without "playing", because we know that the game, even with higher number of piles, has a finite number of states. Anyway, here randomness is key.
   - Methods:
     - `__init__(self) -> None`
     - `training(self) -> None`
     - `play(self, nim: Nim) -> Nimply`

4. **Genotype:**
   - Description: Represents the genotype in the genetic algorithm population.
   - Methods:
     - `__init__(self, s=None, m=None) -> None`
     - `__str__(self) -> str`
     - `__repr__(self) -> str`
     - `init_fitness(self) -> None`
     - `fitness(self, idx) -> None`
     - `score(self) -> None`

5. **Agent_II:**
   - Description: Represents an alternative version of the AI agent using genetic algorithms. This agent play with himself for training, but this little boy needs lots of care and time (generations), please treat it well enough even if he will not look like the smartest boy!
   - Methods:
     - `_select_parent(population) -> None`
     - `_game(self, nim, population, states_0, states_1, moves_0, moves_1, player=0) -> tuple`
     - `_mutation(self, ind: Genotype) -> Genotype`, here there is a chance of a simple mutation or for a simple gaussian mutation
     - `_crossover(self, ind1: Genotype, ind2: Genotype) -> Genotype`, in this case by simply splitting in half
     - `training(self) -> None`
     - `play(self, nim: Nim) -> Nimply`

### Functions

1. `generate_move(s) -> Nimply`
   - Description: Generates a move based on the current state of the game.

2. `generate_state(prev=None) -> tuple`
   - Description: Generates a new game state based on the previous state.

3. `generate_state_list() -> List[tuple]`
   - Description: Generates a list of game states for initializing the population.

4. `generate_population() -> Population`
   - Description: Generates an initial population of Individuals.

5. `select_parent(population) -> Individual`
   - Description: Selects a parent from the population using tournament selection.

6. `mutation(ind: Individual) -> Individual`
   - Description: Performs mutation on an individual.

7. `crossover(inds: List[Individual]) -> Individual`
   - Description: Performs crossover between two individuals.

8. `is_iterable(obj) -> bool`
   - Description: Checks if an object is iterable.

9. `main() -> None`
   - Description: The main function to start the training process.



## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.


