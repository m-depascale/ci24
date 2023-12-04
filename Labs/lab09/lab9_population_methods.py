import random
from copy import copy

def initialize_population(size, fitness):
    pop = []
    for _ in range(size):
        ind = random.choices([0, 1], k=1000)
        score = fitness(ind)
        pop.append(tuple((ind, score)))
    pop.sort(reverse=True, key= lambda ind: ind[1])
    print(f'Initialized population with highest score of an individual of: {pop[0][1]:.2f}')
    return pop

def tournament_selection(population):
    pool = population[:7] + population[-3:]
    return pool

def roulette_wheel_selection(population):
    total_fitness = sum(ind[1] for ind in population)
    selection_probabilities = [ind[1] / total_fitness for ind in population]
    
    selected_indices = []
    for _ in range(len(population)):
        rand_value = random.uniform(0, 1)
        cumulative_prob = 0
        for i, prob in enumerate(selection_probabilities):
            cumulative_prob += prob
            if rand_value <= cumulative_prob:
                selected_indices.append(i)
                break
    
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals

def random_mutation(ind, fitness):
    mutated = ind[0]
    indexes = [random.randint(0, 999) for _ in range(int(1000*0.1))]
    for index in indexes:
        if (random.random() < 0.2):
            mutated[index] = abs(1 - mutated[index])
    score = fitness(mutated)
    return tuple((mutated, score))

def random_mutation(ind, fitness):
    mutated = ind[0]
    indexes = [random.randint(0, 999) for _ in range(int(1000*0.15))]
    for index in indexes:
        if (random.random() < 0.3):
            mutated[index] = 1 - mutated[index]
    score = fitness(mutated)
    return tuple((mutated, score))

def random_mutation_dense(ind):
    mutated = ind[0]
    indexes = [random.randint(0, 999) for _ in range(int(1000*P_MUTATION))]
    for index in indexes:
        rnd = random.random()
        if (rnd < P_MUTATION):
            mutated[index] = abs(1 - mutated[index])
            if random.random() < rnd and (index - 1 > 0):
                mutated[index - 1] = 1 - mutated[index - 1]
            if random.random() < rnd and (index + 1 < 1000):
                mutated[index + 1] = 1 - mutated[index + 1]
    score = fitness(mutated)
    return tuple((mutated, score))

def gaussian_mutation(ind, fitness, mutation_strength=0.2):
    mutated = [bit + random.gauss(0, mutation_strength) for bit in ind[0]]
    mutated = [1 if bit >= 0.5 else 0 for bit in mutated]
    score = fitness(mutated)
    return tuple((mutated, score))

def random_gaussian_mutation(ind, fitness, mutation_strength=0.2):
    mutated = ind[0]
    n = random.randint(0, 990)
    for index in range(n, n+10):
        mutated[index] += random.gauss(0, mutation_strength)
        mutated[index] = 1 if mutated[index] >= 0.5 else 0
    score = fitness(mutated)
    return tuple((mutated, score))


def single_point_crossover(parent1, parent2, fitness):
    n = random.randint(0, 999)
    child = parent1[0][:n] + parent2[0][n:]
    score = fitness(child)
    return tuple((child, score))

def uniform_crossover(parent1, parent2, fitness):
    child = [parent1[0][i] if random.random() < 0.5 else parent2[0][i] for i in range(len(parent1[0]))]
    score = fitness(child)
    return tuple((child, score))

def n_point_crossover(genitore1, genitore2, fitness):
    n = random.randint(0,999)
    l = random.randint(2, max(1000 - n, 10))
    child = genitore1[0][:n] + genitore2[0][n:n+l] + genitore1[0][n+l:]
    if len(child)>1000:
        print("MORE")
        pass
    score = fitness(child)
    return tuple((child, score))

def gm_two_point_crossover(genitore1, genitore2):
    n = random.randint(0,999)
    l =  max(1000 - n, int(1000*P_MUTATION*P_MUTATION))
    child = genitore1[0][:n] + genitore2[0][n:]
    piece = child[n:n+l]
    mutated = [bit + random.gauss(0, 0.2) for bit in piece]
    mutated = [1 if bit >= 0.5 else 0 for bit in mutated]
    child = child[:n] + mutated + child[n+l:]
    if len(child)>1000:
        print("MORE")
        pass
    score = fitness(child)
    return tuple((child, score))

def hamming_distance(a, b):
    h_d = 0
    for bit1, bit2 in zip (a,b):
        if bit1 != bit2: 
            h_d += 1
    return h_d 

def denominator(population):
    denom = []
    sigma_share = 5
    alpha = 1.1
    for i in population:
        sh = 0
        for j in population:
            hd = hamming_distance(i, j)
            if hd < sigma_share:
                sh += 1 - (hd / sigma_share)**alpha
        denom.append(sh)
    return denom

def fitness_sharing(population):
    fitnesses = []
    den = denominator(population)
    for i, ind in enumerate(population):
        fitnesses.append(ind[1] / den[i])
    return fitnesses

def cheat_mutation(ind):
    mutated = ind[0]
    indexes = [random.randint(0, 999) for _ in range(int(1000*P_MUTATION*P_MUTATION))]
    for index in indexes:
        rnd = random.random()
        if (rnd < P_MUTATION):
            mutated[index] = 1 - mutated[index]
            if random.random() < rnd and (index + 1 < 1000) and mutated[index] != mutated[index + 1]:
                mutated[index + 1] = 1 - mutated[index + 1]
    score = fitness(mutated)
    return tuple((mutated, score))

def children_tournament(offspring):
    os = copy(offspring)
    os.sort(reverse=True, key= lambda ind: ind[1])
    l = len(os)
    pool = os[:int(l//10)] + os[-3:] 
    pool.append(os[random.randint(0,len(offspring) - 1)])
    return pool

def chunk_mutation(ind, fitness, mutation_strength=0.4):
    n = random.randint(0, 990)
    chunk = ind[0][n:n+9]
    mutation = [bit + random.gauss(0, mutation_strength) for bit in chunk]
    mutation = [1 if bit >= 0.5 else 0 for bit in mutation]
    mutated = ind[0][:n] + mutation + ind[0][n+9:]
    score = fitness(mutated)
    return tuple((mutated, score))

def chunk_mutation_v2(ind, fitness, mutation_strength=0.2):
    n = random.randint(0, 996)
    chunk = ind[0][n:n+3]
    mutation = [chunk[0] + random.gauss(0, mutation_strength) for _ in range(4)]
    mutation = [1 if bit >= 0.5 else 0 for bit in mutation]
    mutated = ind[0][:n-1] + mutation + ind[0][n+3:]
    score = fitness(mutated)
    return tuple((mutated, score))

def local_search_ea_generic(generations, size, mutation_fun, xover_fun, population, fitness, comma_strategy=False, allopatric_selection=False, default_mode=0, sharing=False):
    pop = copy(population)

    def recombination_only(parents, fitness):
        p1, p2 = random.sample(parents, 2)
        return xover_fun(p1, p2, fitness)
    
    def mutation_only(parents, fitness):
        p1 = random.sample(parents, 1)[0]
        return mutation_fun(p1, fitness)
    
    def xover_mutation(parents, fitness):
        p1, p2 = random.sample(parents, 2)
        return mutation_fun(xover_fun(p1, p2, fitness), fitness)

    def xover_plus_mutation(parents, fitness):
        p1, p2 = random.sample(parents, 2)
        c1 = mutation_fun(xover_fun(p1, p2, fitness), fitness)
        c2 = xover_fun(p1, p2, fitness)
        return [c1, c2]
    
    def switch_case(mode):
        switch_dict = {
            0: xover_mutation,
            1: mutation_only,
            2: recombination_only,
            3: xover_plus_mutation
        }
        return switch_dict.get(mode)
    mode = default_mode
    
    for generation in range(generations):
        if random.random() < 0.1:
            mode = random.randint(0,3)
        else:
            mode = default_mode
        strategy = switch_case(mode)
        
        if sharing:
            adjusted_pop = []
            fitnesses = fitness_sharing(pop)
            for i, ind in enumerate(pop):
                adjusted_pop.append(tuple((ind[0], fitnesses[i])))
            selected_parents = tournament_selection(adjusted_pop)
        else:
            selected_parents = tournament_selection(pop)

        offsprings = []
        for _ in range(0, size):
            if mode==3:
                offsprings.extend(strategy(selected_parents, fitness))
            else:
                offsprings.extend([strategy(selected_parents, fitness)])

        if allopatric_selection:
            pool = children_tournament(offsprings)
            offsprings = random.sample(pool, int(size//10))

        if not comma_strategy:
            pop.extend(offsprings)
            pop.sort(reverse=True, key= lambda ind: ind[1])
            pop = pop[:size]
        else:
            offsprings.sort(reverse=True, key= lambda ind: ind[1])
            pop = offsprings
        if any(ind[1] == 1. for ind in pop):
            print(f'Solution found in {generation + 1} generations with a total of {fitness.calls} fitness calls!')
            break
        else:
            print(f'Solution not found in generation {generation + 1}; best score: {pop[0][1]:.2%}')
    print(f'{pop[0][1]:.2%}')

def local_search_ea_gm_lr(generations, size, xover_fun, population, fitness, comma_strategy=False, allopatric_selection=False, default_mode=0, sharing=False):
    pop = copy(population)

    def linear_decay(initial_alpha, final_alpha, total_generations, current_generation):
        decay_rate = (initial_alpha - final_alpha) / (total_generations*2)
        alpha = initial_alpha - decay_rate * current_generation
        return max(alpha, final_alpha)

    def recombination_only(parents, fitness, alpha):
        p1, p2 = random.sample(parents, 2)
        return xover_fun(p1, p2, fitness)
    
    def mutation_only(parents, fitness, alpha, inner=False):
        if not inner:
            p1 = random.sample(parents, 1)[0]
        else:
            p1 = parents
        #print(p1)
        mutated = [bit + random.gauss(0, alpha) for bit in p1[0]]
        mutated = [1 if bit >= 0.5 else 0 for bit in mutated]
        score = fitness(mutated)
        return tuple((mutated, score))
    
    def xover_mutation(parents, fitness, alpha):
        p1, p2 = random.sample(parents, 2)
        return mutation_only(xover_fun(p1, p2, fitness), fitness, alpha, inner=True)

    def xover_plus_mutation(parents, fitness, alpha):
        p1, p2 = random.sample(parents, 2)
        c1 = mutation_only(xover_fun(p1, p2, fitness), fitness, alpha, inner=True)
        c2 = xover_fun(p1, p2, fitness)
        return [c1, c2]
    
    def switch_case(mode):
        switch_dict = {
            0: xover_mutation,
            1: mutation_only,
            2: recombination_only,
            3: xover_plus_mutation
        }
        return switch_dict.get(mode, mutation_only)
    mode = default_mode

    for generation in range(generations):
        alpha = linear_decay(0.4, 0.1, generations, generation)
        if random.random() < 0.05:
            mode = random.randint(0,3)
        else:
            mode = default_mode
        strategy = switch_case(mode)
        
        if sharing:
            adjusted_pop = []
            fitnesses = fitness_sharing(pop)
            for i, ind in enumerate(pop):
                adjusted_pop.append(tuple((ind[0], fitnesses[i])))
            selected_parents = tournament_selection(adjusted_pop)
        else:
            selected_parents = tournament_selection(pop)

        offsprings = []
        for _ in range(0, size):
            if mode==3:
                offsprings.extend(strategy(selected_parents, fitness, alpha))
            else:
                offsprings.extend([strategy(selected_parents, fitness, alpha)])

        if allopatric_selection:
            pool = children_tournament(offsprings)
            offsprings = random.sample(pool, int(size//10))

        if not comma_strategy:
            pop.extend(offsprings)
            pop.sort(reverse=True, key= lambda ind: ind[1])
            pop = pop[:size]
        else:
            offsprings.sort(reverse=True, key= lambda ind: ind[1])
            pop = offsprings
        if any(ind[1] == 1. for ind in pop):
            print(f'Solution found in {generation + 1} generations with a total of {fitness.calls} fitness calls!')
            break
        else:
            print(f'Solution not found in generation {generation + 1}; best score: {pop[0][1]:.2%}')
    print(f'{pop[0][1]:.2%}')

def local_search_t_gm_npc(generations, size, population, fitness, comma_strategy=False, allopatric_selection=False, default_mode=0, sharing=False):
    pop = copy(population)
    for generation in range(generations):
        selected_parents = tournament_selection(pop)

        new_population = []
        for _ in range(0, size, 2):
            genitore1, genitore2 = random.sample(selected_parents, 2)
            child1 = n_point_crossover(genitore1, genitore2, fitness)
            child2 = gaussian_mutation(n_point_crossover(genitore1, genitore2, fitness), fitness)
            new_population.extend([child1, child2])
       
        if comma_strategy:
            new_population.sort(reverse=True, key= lambda ind: ind[1])
            pop = new_population
        else:
            if allopatric_selection:
                new_population = children_tournament(new_population)
            pop.extend(new_population)
            pop.sort(reverse=True, key= lambda ind: ind[1])
            pop = pop[:size]

        if any(ind[1] == 1. for ind in pop):
            print(f'Solution found in {generation + 1} generations with a total of {fitness.calls} fitness calls!')
            break
        else:
            print(f'Solution not found in generation {generation + 1}; best score: {pop[0][1]:.2%}')
        
    print(f'{pop[0][1]:.2%}')

def local_search_bitflip_sc(generations, size, population, fitness, comma_strategy=False, allopatric_selection=False, default_mode=0, sharing=False):
    pop = copy(population)
    alpha = 0.2
    for generation in range(generations):
        selected_parents = tournament_selection(pop)
        new_population = []
        if generation%10:
            alpha = max(alpha - 0.02, 0.1)
        if generation%150:
            alpha = 0.2
        for _ in range(0, size):
            if random.random() < 0.3:
                genitore1, genitore2 = random.sample(selected_parents, 2)
                child = single_point_crossover(genitore1, genitore2, fitness)
            else:
                p = random.sample(selected_parents, 1)[0]
                child = gaussian_mutation(p, fitness, 0.05)
            new_population.extend([child])
       
        if comma_strategy:
            new_population.sort(reverse=True, key= lambda ind: ind[1])
            pop = new_population
        else:
            if allopatric_selection:
                new_population = children_tournament(new_population)
            pop.extend(new_population)
            pop.sort(reverse=True, key= lambda ind: ind[1])
            pop = pop[:size]

        if any(ind[1] == 1. for ind in pop):
            print(f'Solution found in {generation + 1} generations with a total of {fitness.calls} fitness calls!')
            break
        else:
            print(f'Solution not found in generation {generation + 1}; best score: {pop[0][1]:.2%}')
        
    print(f'{pop[0][1]:.2%}')