import random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from deap import base, creator, tools, algorithms
from time import sleep

# Load dataset (e.g., Iris dataset)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define Genetic Algorithm Parameters
POP_SIZE = 20      # Population size
MAX_GEN = 10       # Maximum generations
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5

# Define bounds for hyperparameters
param_bounds = {
    'C': (0.1, 100),       # Regularization parameter for SVM (Positive range)
    'gamma': (0.0001, 1),  # Ensure gamma is always a positive float
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Different kernels for SVM
}

# Define the fitness function (accuracy in this case)
def evaluate_individual(individual):
    C = max(0.1, individual[0])  # Ensure C is positive
    gamma = max(0.0001, individual[1])  # Ensure gamma is positive
    kernel_idx = individual[2]
    kernel = param_bounds['kernel'][int(kernel_idx)]
    
    # Define the model
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    
    # Evaluate model using cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Return the mean of cross-validation scores as fitness value
    return (np.mean(scores),)

# Define the individual (one possible solution)
def create_individual():
    C = random.uniform(*param_bounds['C'])  # Random C value in the given range
    gamma = random.uniform(0.0001, 1)  # Ensure gamma is always positive
    kernel_idx = random.randint(0, len(param_bounds['kernel']) - 1)  # Random kernel index
    return [C, gamma, kernel_idx]

# Define mutation (randomly modify parameters)
def mutate_individual(individual):
    # Randomly mutate one of the hyperparameters
    if random.random() < MUTATION_RATE:
        individual[0] = random.uniform(*param_bounds['C'])  # Mutate C, ensure it's positive
    if random.random() < MUTATION_RATE:
        individual[1] = random.uniform(0.0001, 1)  # Ensure gamma is always positive
    if random.random() < MUTATION_RATE:
        individual[2] = random.randint(0, len(param_bounds['kernel']) - 1)  # Mutate kernel
    
    # Return the individual in a tuple
    return (individual,)

# Create the genetic algorithm structure using DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # We want to maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_individual)

# Main optimization function
def optimize():
    # Create the initial population
    population = toolbox.population(n=POP_SIZE)

    # Apply the genetic algorithm
    hof = tools.HallOfFame(1)  # Keep track of the best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, ngen=MAX_GEN,
        stats=stats, halloffame=hof, verbose=True
    )

   
 # Print the best individual found
    best_individual = hof[0]
    print(f"\nBest individual: C={best_individual[0]}, gamma={best_individual[1]}, kernel={param_bounds['kernel'][int(best_individual[2])]}")
    print(f"Best fitness (accuracy): {best_individual.fitness.values[0]}")

# Run the optimization
if __name__ == "__main__":
    optimize()
    sleep(100)
