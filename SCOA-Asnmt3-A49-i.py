import random
import string
from time import sleep

class GeneticAlgorithm:
    def __init__(self, target, population_size, mutation_rate, max_generations):
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.gene_pool = string.ascii_letters + ' '  # Allow letters and space as possible characters
        self.target_length = len(target)
        self.population = self._init_population()

    def _init_population(self):
        """ Initialize the population with random words of the target length. """
        population = []
        for _ in range(self.population_size):
            individual = ''.join(random.choice(self.gene_pool) for _ in range(self.target_length))
            population.append(individual)
        return population

    def _fitness(self, individual):
        """ Calculate fitness by comparing each character of the individual to the target. """
        fitness = 0
        for i in range(len(individual)):
            if individual[i] == self.target[i]:
                fitness += 1
        return fitness

    def _mutate(self, individual):
        """ Randomly mutate an individual's genes based on the mutation rate. """
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.gene_pool)
        return ''.join(individual)

    def _crossover(self, parent1, parent2):
        """ Perform crossover between two parents to create a child. """
        child = []
        for i in range(self.target_length):
            if random.random() > 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return ''.join(child)

    def _select_parents(self):
        """ Select two parents using tournament selection. """
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        parent1 = max(tournament, key=self._fitness)
        parent2 = max(random.sample(self.population, tournament_size), key=self._fitness)
        return parent1, parent2

    def evolve(self):
        """ Run the genetic algorithm to evolve the population toward the target word. """
        for generation in range(self.max_generations):
            # Calculate fitness for the entire population
            fitness_scores = [(individual, self._fitness(individual)) for individual in self.population]
            best_individual, best_fitness = max(fitness_scores, key=lambda x: x[1])

            # Print the best result of this generation
            print(f"Generation {generation + 1}: Best Word: {best_individual}, Fitness: {best_fitness}")

            # Check if the target word has been found
            if best_individual == self.target:
                print(f"\nTarget word '{self.target}' found in {generation + 1} generations!")
                break

            # Create a new population using crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self._select_parents()
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population
        else:
            print(f"\nMaximum generations reached. Best word: {best_individual} with fitness: {best_fitness}")


# Example Usage
if __name__ == "__main__":
    target_word = "This is a code"
    population_size = 100
    mutation_rate = 0.01
    max_generations = 1000

    ga = GeneticAlgorithm(target_word, population_size, mutation_rate, max_generations)
    ga.evolve()
    sleep(100)
