import random
from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from deap import algorithms, base, creator, tools

matplotlib.use('GTK3Agg')

# random.seed(42)

# min
# 49 .. 112
# Equation: f(x) = -1 - 4*x - 7**2 + 7**3


POPULATION_SIZE = 200
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 50
MUT_FLIP_BIT_INDPB = 0.05
MIN_VALUE = 49
MAX_VALUE = 112

def equation(x):
    return -1 - 4 * x - 7**2 + 7**3

def bin_to_decimal(bits):
    return int(''.join(str(bit) for bit in bits), 2)

def decimal_to_bin(decimal):
    return [int(x) for x in bin(decimal)[2:].zfill(7)]


def evaluate(individual):
    decimal_x = bin_to_decimal(individual)
    fitness_value = equation(decimal_x)
    return fitness_value,


def gen_bits(a , b):
    decimal = random.randint(a, b)
    binary = decimal_to_bin(decimal)
    return binary

gen_bits = partial(gen_bits, MIN_VALUE, MAX_VALUE)


def cx_one_point_in_range(ind1, ind2):
    while True:
        ind1, ind2 = tools.cxOnePoint(ind1, ind2)
        if MIN_VALUE <= bin_to_decimal(ind1) <= MAX_VALUE \
               and MIN_VALUE <= bin_to_decimal(ind2) <= MAX_VALUE:
            break

    return ind1, ind1


def mut_flip_bit_in_range(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            while True:
                individual = tools.mutFlipBit(individual, indpb)[0]

                decimal = bin_to_decimal(individual)
                if MIN_VALUE <= decimal <= MAX_VALUE:
                    break

    return individual,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, gen_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", cx_one_point_in_range)
toolbox.register("mutate", mut_flip_bit_in_range, indpb=MUT_FLIP_BIT_INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=POPULATION_SIZE)

# Perform the evolution
max_fitness_values = []
mean_fitness_values = []
for i, generation in enumerate(range(MAX_GENERATIONS)):
    fitness_values = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitness_values):
        individual.fitness.values = fitnessValue

    fitness_values = [
        individual.fitness.values[0]
        for individual in population
    ]

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTATION_PROBABILITY:
            toolbox.mutate(mutant)

            del mutant.fitness.values

    fresh_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fresh_fitness_values = list(map(toolbox.evaluate, fresh_individuals))

    for individual, fitness_value in zip(fresh_individuals, fresh_fitness_values):
        individual.fitness.values = fitness_value

    population[:] = offspring

    max_fitness = min(fitness_values)
    mean_fitness = sum(fitness_values) / len(population)
    max_fitness_values.append(max_fitness)
    mean_fitness_values.append(mean_fitness)

    print(f"- Generation {i}: max fitness = {max_fitness}, mean fitness = {mean_fitness}")

best_individual = tools.selBest(population, k=1)[0]
print("Best Individual:", bin_to_decimal(best_individual))
print("Fitness Value:", best_individual.fitness.values[0])

plt.plot(max_fitness_values, color="blue")
plt.plot(mean_fitness_values, color="green")
plt.xlabel("Generation")
plt.ylabel("Best/Mean Fitness Value")
plt.title("Genetic Algorithm - Best Fitness Value Evolution")
plt.show()
