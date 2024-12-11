import numpy as np
from time import sleep

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            print(f"\n--- Iteration {i+1} ---")
            all_paths = self.gen_all_paths()
            # Find the n_best (optimal) paths
            sorted_paths = sorted(all_paths, key=lambda x: x[1])[:self.n_best]
            self.spread_pheromone(sorted_paths)
            # Print only the best paths
            print(f"\nOptimal paths in this iteration (Top {self.n_best}):")
            for idx, (path, dist) in enumerate(sorted_paths):
                print(f"Ant {idx+1}: Path: {path} with distance: {dist}")
            # Track the shortest path
            shortest_path = min(sorted_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= (1 - self.decay)
        print(f"\nAll-time shortest path: {all_time_shortest_path[0]} with distance: {all_time_shortest_path[1]}")
        return all_time_shortest_path

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path)):
            dist = self.distances[path[i-1]][path[i]]
            if dist != np.inf:  # Ensure we are not adding infinite values
                total_dist += dist
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            path_dist = self.gen_path_dist(path)
            all_paths.append((path, path_dist))
        return all_paths

    def gen_path(self, start):
        path = [start]
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        path.append(start)  # Return to start point
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0  # Remove pheromones for visited cities

        # Calculate probability based on pheromone and heuristic (1/distance)
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        if row.sum() == 0:
            # If all probabilities are zero, choose a random unvisited city
            available_moves = list(set(self.all_inds) - visited)
            move = np.random.choice(available_moves)
            return move
        else:
            norm_row = row / row.sum()
            return np.random.choice(self.all_inds, 1, p=norm_row)[0]

    def spread_pheromone(self, best_paths):
        print("\nSpreading pheromones on best paths:")
        for path, dist in best_paths:
            print(f"Path: {path} with distance: {dist}")
            for move in zip(path[:-1], path[1:]):
                self.pheromone[move] += 1.0 / dist  # Spread pheromone inversely proportional to distance
        print(f"Pheromone matrix after update:\n{self.pheromone}")

# Example of usage
if __name__ == "__main__":
    distances = np.array([[np.inf, 2, 2, 5, 7],
                          [2, np.inf, 4, 8, 2],
                          [2, 4, np.inf, 1, 3],
                          [5, 8, 1, np.inf, 2],
                          [7, 2, 3, 2, np.inf]])

    ant_colony = AntColony(distances, 10, 5, 100, 0.95, alpha=1, beta=2)
    shortest_path = ant_colony.run()
    print(f"\nFinal shortest path: {shortest_path[0]} with distance: {shortest_path[1]}")
    sleep(100)
