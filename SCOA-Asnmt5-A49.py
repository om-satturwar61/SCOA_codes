import numpy as np
import random
from time import sleep

class PSOClustering:
    def __init__(self, data, num_clusters, num_particles, max_iter, inertia=0.7, cognitive=1.5, social=1.5):
        self.data = data
        self.num_clusters = num_clusters
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia  # w
        self.cognitive = cognitive  # c1
        self.social = social  # c2
        self.num_features = data.shape[1]  # Dimensionality of the data
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.particles = []

    class Particle:
        def __init__(self, position, velocity, best_position, best_score):
            self.position = position  # Cluster centroids
            self.velocity = velocity
            self.best_position = best_position  # Best known position of this particle
            self.best_score = best_score  # Best known score of this particle

    def initialize_particles(self):
        for _ in range(self.num_particles):
            # Initialize particle's position (random centroids) and velocity
            position = np.array([random.choice(self.data) for _ in range(self.num_clusters)])
            velocity = np.random.rand(self.num_clusters, self.num_features)
            best_position = np.copy(position)
            best_score = self.evaluate(position)
            particle = self.Particle(position, velocity, best_position, best_score)
            self.particles.append(particle)
            # Update global best position if necessary
            if best_score < self.global_best_score:
                self.global_best_position = np.copy(best_position)
                self.global_best_score = best_score

    def evaluate(self, centroids):
        """ Calculate the clustering error (within-cluster sum of squared errors). """
        total_distance = 0
        for point in self.data:
            # Find the nearest centroid for each data point
            distances = np.linalg.norm(point - centroids, axis=1)
            min_distance = np.min(distances)
            total_distance += min_distance ** 2
        return total_distance

    def update_velocity(self, particle):
        """ Update particle velocity based on inertia, cognitive and social components. """
        r1, r2 = np.random.rand(), np.random.rand()  # Random coefficients
        cognitive_velocity = self.cognitive * r1 * (particle.best_position - particle.position)
        social_velocity = self.social * r2 * (self.global_best_position - particle.position)
        particle.velocity = self.inertia * particle.velocity + cognitive_velocity + social_velocity

    def update_position(self, particle):
        """ Update particle's position (centroids) based on its velocity. """
        particle.position += particle.velocity

    def optimize(self):
        self.initialize_particles()

        for iteration in range(self.max_iter):
            for particle in self.particles:
                # Update the particle's velocity and position
                self.update_velocity(particle)
                self.update_position(particle)

                # Evaluate the particle's new position
                current_score = self.evaluate(particle.position)

                # Update the particle's personal best position if necessary
                if current_score < particle.best_score:
                    particle.best_position = np.copy(particle.position)
                    particle.best_score = current_score

                # Update the global best position if necessary
                if current_score < self.global_best_score:
                    self.global_best_position = np.copy(particle.position)
                    self.global_best_score = current_score

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score

    def predict(self, data):
        """ Assigns each data point to the nearest cluster centroid. """
        clusters = []
        for point in data:
            distances = np.linalg.norm(point - self.global_best_position, axis=1)
            clusters.append(np.argmin(distances))
        return np.array(clusters)

# Example Usage:
if __name__ == "__main__":
    # Sample data (e.g., 2D data points)
    data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])

    # Parameters: number of clusters, number of particles, maximum iterations
    num_clusters = 3
    num_particles = 10
    max_iter = 100

    pso_clustering = PSOClustering(data, num_clusters, num_particles, max_iter)
    best_centroids, best_score = pso_clustering.optimize()

    print("\nOptimal Cluster Centroids:\n", best_centroids)
    print("\nFinal Best Score (Sum of Squared Errors):", best_score)

    # Predict the clusters for the data
    clusters = pso_clustering.predict(data)
    print("\nCluster Assignments for Data Points:\n", clusters)
    sleep(100)