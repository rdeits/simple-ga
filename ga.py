from __future__ import division

import numpy as np
import random
import json

"""
This file implements a basic genetic algorithm optimizer for an arbitrary
optimization problem containing an objective function which you would like to
minimize.
"""


class FitnessFunction:
    """
    Contains an objective function along with its input upper and lower bounds.
    The objective function is called on the genotype array of an Individual when
    that individual's fitness is accessed.
    """
    def __init__(self, obj_fun, num_vars, lb=None, ub=None):
        # function should take a vector of length num_vars
        self._function = obj_fun
        self.num_vars = num_vars
        if lb is None:
            self.lb = [0 for i in range(num_vars)]
        else:
            self.lb = lb
        if ub is None:
            self.ub = [1 for i in range(num_vars)]
        else:
            self.ub = ub

    def __call__(self, x):
        assert len(x) == self.num_vars, "len(x) != num_vars"
        assert [x[i] > self.lb[i] for i in range(self.num_vars)], "x below lb"
        assert [x[i] < self.ub[i] for i in range(self.num_vars)], "x above ub"
        return self._function(x)

    def plot_estimate(self, x):
        pass


class GA(object):
    """A genetic algorithm class. The only required argument is a fitness
    function, which takes the form of a FitnessFunction object here. That
    object must have fields lb, ub, and num_vars and a __call__ method which
    expects a list or array of length num_vars.

    Optional parameters:
    pop_size (int): number of individuals in the population
    keep_fraction (float): fraction of the population that will remain after culling
    mut_rate (float): the likelihood that an individual element of each genome will be mutated
    elite_count (int): the number of top individuals preserved unmutated in each generation
    max_generations (int): maximum number of generations before stopping the algorithm
    min_fitness (float): fitness value at which the algorithm will stop
    stall_generations (int): number of generations over which less than 5% improvement in best fitness will be considered a stall (at which point the algorithm will stop)
    verbose (bool): print additional debugging information
    """

    @staticmethod
    def load(fitness_function, fname='ga_state.json'):
        loaded_data = json.load(open(fname, 'r'))
        return GA(fitness_function, **loaded_data)

    def __init__(self, fitness_function,
            pop_size=12,
            keep_fraction=.5,
            mut_rate=0.1,
            elite_count=1,
            max_generations=1000,
            min_fitness=0,
            stall_generations=10,
            verbose=False,
            history=[],
            save_state=False):
        self.fitness_function = fitness_function
        self.pop_size = pop_size
        self.keep_fraction = keep_fraction
        self.mut_rate = mut_rate
        self.elite_count = elite_count
        self.max_generations = max_generations
        self.min_fitness = min_fitness
        self.stall_generations = stall_generations
        self.keep_num = int(self.keep_fraction * pop_size)
        self.verbose = verbose
        self.lb = self.fitness_function.lb
        self.ub = self.fitness_function.ub
        self.num_vars = self.fitness_function.num_vars

        ### Load the state from a previous run, if given ###
        self.history = history
        self.generation = len(history)
        self.best_fitnesses = map(lambda x: min(y['fitness'] for y in x),
                                  history)
        if self.history != []:
            self.individuals = [Individual(self.fitness_function,
                                           indv['genotype'],
                                           indv['fitness'])\
                                for indv in history[-1]]
            self.update_population()
        else:
            self.create_population(self.pop_size)

        # TODO: save_state could hold the filename, with None indicating no saving
        self.save_state = save_state

    def create_population(self, size):
        """Initialize the population with new individuals.
        The generation of a new (random) individual is handled
        by the Individual class."""
        self.individuals = [Individual(self.fitness_function, None)\
                for i in range(size)]
        if self.verbose:
            print "Ceated initial pop."
            print self.individuals

    def save(self, fname='ga_state.json'):
        vars_to_save = ['pop_size', 'keep_fraction', 'mut_rate',
                        'elite_count', 'max_generations', 'min_fitness',
                        'stall_generations', 'verbose', 'history',
                        'save_state']
        data_to_save = {x: self.__dict__[x] for x in vars_to_save}
        json.dump(data_to_save, open(fname, 'w'), indent=2, sort_keys=True)

    def sort(self):
        """Sort the population"""
        self.individuals.sort(key=lambda x: x.fitness)

    def evaluate(self):
        """Evaluate each individual's fitness function"""
        # This is actually unnecessary. Because of the way Individual.fitness
        # is set up as a Python property, the call to self.sort() will
        # actually call the fitness function as necessary. This is just done
        # to be more explicit and clear.
        for indiv in self.individuals:
            indiv.evaluate()

    def step(self):
        '''Run an entire generation. Returns True if the algorithm is finished, False otherwise.'''
        if self.verbose:
            print "stepping"
        self.evaluate()
        self.sort()
        self.history.append([{'genotype': indiv.genotype,
                                          'fitness': indiv.fitness} for
                                         indiv in self.individuals])
        # After sorting, self.individuals[0] is the most fit individual
        self.best_fitnesses.append(self.individuals[0].fitness)
        if self.verbose:
            self.print_status()
        if self.save_state:
            self.save()
        if self.done():
            self.report()
            return True
        self.update_population()
        self.generation += 1
        return False

    def update_population(self):
        self.cull()
        self.reproduce()
        self.mutate_all()

    def print_status(self):
        print "=================================="
        print "Generation", self.generation, "run. Sorted individuals:"
        for indiv in self.individuals:
            print "Genotype:", indiv.genotype
            print "Fitness:", indiv.fitness, "\n"

    def run(self):
        try:
            while not self.step():
                pass
        except KeyboardInterrupt:
            # Exit gracefully on ^c
            pass
        self.sort()
        return self.individuals[0].fitness, self.individuals[0].genotype, self.history

    def done(self):
        return (self.generation >= self.max_generations or
                (self.generation > self.stall_generations and
                    (self.best_fitnesses[-self.stall_generations] -
                        self.best_fitnesses[-1] <= 0.05 * self.best_fitnesses[-1])) or
                    (self.best_fitnesses[-1] <= self.min_fitness))

    def report(self):
        self.sort()
        print "Best fitness:", self.individuals[0].fitness
        print "Best genotype:", self.individuals[0].genotype

    def reproduce(self):
        new_children = []
        assert (self.pop_size - len(self.individuals)) % 2 == 0
        for i in range((self.pop_size - len(self.individuals)) // 2):
            [parent0, parent1] = self.select_parents()
            [child0, child1] = self.combine(parent0, parent1)
            new_children.append(child0)
            new_children.append(child1)
            # print "combined genotypes:"
            # print parent0.genotype
            # print parent1.genotype
            # print "got genotypes:"
            # print child0.genotype
            # print child1.genotype
            # print "\n\n"
        self.individuals = self.individuals + new_children

    def select_parents(self):
        """This function chooses two parents from the population to be  parents
        for two children by selecting the best two out of three random
        individuals from the population."""

        # Choose three different individuals at random
        candidates = random.sample(self.individuals, 3)

        # choose the best two candidates to be parents and return
        # those individuals
        candidates.sort(key=lambda x: x.fitness)
        return [candidates[0], candidates[1]]

    def combine(self, parent0, parent1):
        """Combine the genotypes of two parents to yield two child genotypes"""
        crossover_point = random.randint(0, self.num_vars - 1)
        genotype0 = (parent0.genotype[:crossover_point] +
                parent1.genotype[crossover_point:])
        genotype1 = (parent1.genotype[:crossover_point] +
                parent1.genotype[crossover_point:])
        return [Individual(self.fitness_function, genotype0),
                Individual(self.fitness_function, genotype1)]

    def cull(self):
        """self.cull() removes the least-fit members of the population, keeping
        self.keep_num individuals"""
        # remove the individuals at the end of the list
        # (those with the highest (worst) fitness values)
        self.sort()
        self.individuals = self.individuals[:self.keep_num]

    def mutate_all(self):
        """perform mutation on all individuals in the population except the top
        self.elite_count by randomly replacing values in those individuals with
        new values within the allowable range"""
        for indiv in self.individuals[self.elite_count:]:
            self.update_mutation_factor()
            new_genotype = indiv.genotype
            for j in range(self.num_vars):
                if random.random() < self.mut_rate:
                    offset = (2 * (random.random() - 0.5)
                            * self.mutation_factor
                            * (self.ub[j] - self.lb[j]))
                    # print "offset:",offset
                    # if offset > 0 and offset < 1:
                        # offset = 1
                    # elif offset < 0 and offset > -1:
                        # offset = -1
                    # else:
                        # offset = int(offset)
                    new_genotype[j] = new_genotype[j] + offset
                    if new_genotype[j] > self.ub[j]:
                        new_genotype[j] = self.ub[j]
                    elif new_genotype[j] < self.lb[j]:
                        new_genotype[j] = self.lb[j]
            indiv.genotype = new_genotype

    def update_mutation_factor(self):
        """The mutation factor scales the amount by which the genotypes are mutated.
        It decreases with the number of generations, starting at generation 10."""
        if self.generation > 10:
            self.mutation_factor = 2.5 / self.generation
        else:
            self.mutation_factor = 0.25


class Individual(object):
    """each individual in the population is an instance of Individual,  which
    contains two public fields:
    self.genotype is a numpy array of the continuous values which that  individual passes to the objective function
    self.fitness is the numerical fitness score for that individual as  returned by the objective function"""
    def __init__(self, fitness_function, genotype=None, fitness=None):
        if genotype is None:
            self.genotype = [(random.random() * (fitness_function.ub[i] -
                fitness_function.lb[i]) + fitness_function.lb[i])\
                        for i in range(fitness_function.num_vars)]
        else:
            self.genotype = genotype
        if fitness is not None:
            self._dirty = False
            self._fitness = fitness
        self.fitness_function = fitness_function

    @property
    def genotype(self):
        return self._genotype

    @genotype.setter
    def genotype(self, value):
        self._genotype = value
        self._dirty = True
        self._fitness = np.inf

    @property
    def fitness(self):
        self.evaluate()
        return self._fitness

    def evaluate(self):
        if self._dirty:
            self._fitness = self.fitness_function(self.genotype)
            self._dirty = False

if __name__ == '__main__':
    func = FitnessFunction(obj_fun=lambda x: np.sum(np.power(x, 2)),
                           num_vars=4,
                           lb=[-10] * 4,
                           ub=[10] * 4)
    ga = GA(fitness_function=func,
            stall_generations=10,
            save_state=True)
    fitness, genotype, history = ga.run()
    print map(lambda x: min(y['fitness'] for y in x), history)
