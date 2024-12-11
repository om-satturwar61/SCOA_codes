import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
from time import sleep

data = load_breast_cancer()
X = data.data
y = data.target

# Split and scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define fuzzy variables
fitness = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'fitness')
mutation_rate = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'mutation_rate')
crossover_rate = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'crossover_rate')

# Define membership functions
fitness['low'] = fuzz.trimf(fitness.universe, [0, 0, 0.5])
fitness['medium'] = fuzz.trimf(fitness.universe, [0, 0.5, 1])
fitness['high'] = fuzz.trimf(fitness.universe, [0.5, 1, 1])

mutation_rate['low'] = fuzz.trimf(mutation_rate.universe, [0, 0, 0.5])
mutation_rate['medium'] = fuzz.trimf(mutation_rate.universe, [0, 0.5, 1])
mutation_rate['high'] = fuzz.trimf(mutation_rate.universe, [0.5, 1, 1])

crossover_rate['low'] = fuzz.trimf(crossover_rate.universe, [0, 0, 0.5])
crossover_rate['medium'] = fuzz.trimf(crossover_rate.universe, [0, 0.5, 1])
crossover_rate['high'] = fuzz.trimf(crossover_rate.universe, [0.5, 1, 1])

# Define fuzzy rules
rules = [
    ctrl.Rule(fitness['low'], mutation_rate['high']),
    ctrl.Rule(fitness['medium'], mutation_rate['medium']),
    ctrl.Rule(fitness['high'], mutation_rate['low']),
    ctrl.Rule(fitness['low'], crossover_rate['low']),
    ctrl.Rule(fitness['medium'], crossover_rate['medium']),
    ctrl.Rule(fitness['high'], crossover_rate['high']),
]

# Control system and simulation
fitness_ctrl = ctrl.ControlSystem(rules)
fitness_simulation = ctrl.ControlSystemSimulation(fitness_ctrl)

# Genetic Algorithm Parameters
population_size = 10
num_generations = 20
param_bounds = {'C': (0.1, 100), 'gamma': (0.001, 1)}

# Initialize population with random hyperparameters
def initialize_population(size, bounds):
    population = []
    for _ in range(size):
        individual = {
            'C': random.uniform(bounds['C'][0], bounds['C'][1]),
            'gamma': random.uniform(bounds['gamma'][0], bounds['gamma'][1])
        }
        population.append(individual)
    return population

# Evaluate fitness (accuracy of SVM with given hyperparameters)
def fitness_score(individual):
    model = SVC(C=individual['C'], gamma=individual['gamma'])
    return cross_val_score(model, X_train, y_train, cv=3).mean()

# Selection, Crossover, Mutation
def select_parents(population, scores):
    parents = random.choices(population, weights=scores, k=2)
    return parents

def crossover(parent1, parent2):
    child = {
        'C': (parent1['C'] + parent2['C']) / 2,
        'gamma': (parent1['gamma'] + parent2['gamma']) / 2
    }
    return child

def mutate(individual, mutation_rate, bounds):
    if random.random() < mutation_rate:
        individual['C'] += random.uniform(-0.1, 0.1) * (bounds['C'][1] - bounds['C'][0])
        individual['gamma'] += random.uniform(-0.01, 0.01) * (bounds['gamma'][1] - bounds['gamma'][0])
        # Ensure bounds are respected
        individual['C'] = np.clip(individual['C'], bounds['C'][0], bounds['C'][1])
        individual['gamma'] = np.clip(individual['gamma'], bounds['gamma'][0], bounds['gamma'][1])
    return individual

# Main Genetic Algorithm loop
population = initialize_population(population_size, param_bounds)

for generation in range(num_generations):
    # Calculate fitness for each individual
    scores = [fitness_score(ind) for ind in population]

    # Update fuzzy logic based mutation and crossover rates based on best score
    best_score = max(scores)
    fitness_simulation.input['fitness'] = best_score
    fitness_simulation.compute()
    mutation_rate = fitness_simulation.output['mutation_rate']
    crossover_rate = fitness_simulation.output['crossover_rate']

    # Generate new population
    new_population = []
    for _ in range(population_size // 2):
        # Selection
        parent1, parent2 = select_parents(population, scores)
        
        # Crossover
        if random.random() < crossover_rate:
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
        else:
            child1, child2 = parent1, parent2

        # Mutation
        child1 = mutate(child1, mutation_rate, param_bounds)
        child2 = mutate(child2, mutation_rate, param_bounds)

        new_population.extend([child1, child2])

    population = new_population

plain_SVM = SVC()
plain_SVM.fit(X_train, y_train)
y_pred_plain = plain_SVM.predict(X_test)
plain_accuracy = accuracy_score(y_test, y_pred_plain)

best_individual = max(population, key=fitness_score)
best_model = SVC(C=best_individual['C'], gamma=best_individual['gamma'])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Plain SVM Accuracy:", plain_accuracy)
print("Optimized SVM Accuracy:", accuracy)
print("Best Hyperparameters:", best_individual)
sleep(100)