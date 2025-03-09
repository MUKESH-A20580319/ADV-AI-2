import sys
import pandas as pd
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt

def euclidean_distance(coord1, coord2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def load_data(file_path):
    """Load CSV file and construct distance matrix."""
    df = pd.read_csv(file_path, header=None, names=['State', 'X', 'Y'])
    states = df['State'].tolist()
    coordinates = dict(zip(states, zip(df['X'], df['Y'])))

    # Create distance matrix
    distance_matrix = {
        state1: {state2: euclidean_distance(coordinates[state1], coordinates[state2]) for state2 in states}
        for state1 in states
    }
    return states, distance_matrix

def initialize_population(states, population_size):
    """Generate an initial population of random permutations."""
    population = []
    for _ in range(population_size):
        individual = states[1:]  # Exclude start node for permutation
        random.shuffle(individual)
        population.append([states[0]] + individual + [states[0]])  # Always start and end at first node
    return population

def fitness(solution, distance_matrix):
    """Calculate fitness as the inverse of total path cost."""
    total_distance = sum(distance_matrix[solution[i]][solution[i+1]] for i in range(len(solution)-1))
    return 1 / total_distance  # Higher fitness for shorter paths

def tournament_selection(population, fitness_values, k=5):
    """Select best individual from a random subset (Tournament Selection)."""
    selected = random.sample(list(zip(population, fitness_values)), k)
    return max(selected, key=lambda x: x[1])[0]

def two_point_crossover(parent1, parent2):
    """Perform Two-Point Crossover (TPX)."""
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(1, size-1), 2))  # Select two crossover points

    child = [None] * size
    child[p1:p2] = parent1[p1:p2]  # Copy middle section from parent1

    # Fill remaining positions with genes from parent2 in order
    p2_elements = [gene for gene in parent2 if gene not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_elements[idx]
            idx += 1
    return child

def swap_mutation(individual, mutation_prob):
    """Apply swap mutation with given probability."""
    if random.random() < mutation_prob:
        i, j = random.sample(range(1, len(individual)-1), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def two_opt(solution, distance_matrix, max_swaps=50):
    """Apply 2-opt local optimization with limited swaps."""
    best = solution
    best_cost = sum(distance_matrix[best[i]][best[i+1]] for i in range(len(best)-1))
    swap_count = 0

    for i in range(1, len(solution) - 2):
        for j in range(i + 1, len(solution) - 1):
            if j - i == 1:  # Skip adjacent swaps
                continue
            new_solution = best[:i] + best[i:j][::-1] + best[j:]
            new_cost = sum(distance_matrix[new_solution[k]][new_solution[k+1]] for k in range(len(new_solution)-1))

            if new_cost < best_cost:
                best = new_solution
                best_cost = new_cost
                swap_count += 1

            if swap_count >= max_swaps:  # Stop early after max swaps
                return best

    return best

def genetic_algorithm(states, distance_matrix, num_iterations, mutation_prob, population_size=500):
    """Run Optimized Genetic Algorithm to solve TSP."""
    population = initialize_population(states, population_size)

    best_solution = None
    best_cost = float('inf')

    iteration_data = []  # To store iteration-wise results for plotting
    initial_solution = population[0]  # Store the first generated random solution

    for iteration in range(num_iterations):
        fitness_values = [fitness(ind, distance_matrix) for ind in population]
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = tournament_selection(population, fitness_values), tournament_selection(population, fitness_values)
            child1, child2 = two_point_crossover(parent1, parent2), two_point_crossover(parent2, parent1)
            child1, child2 = swap_mutation(child1, mutation_prob), swap_mutation(child2, mutation_prob)

            # Apply local optimization (2-opt)
            child1 = two_opt(child1, distance_matrix)
            child2 = two_opt(child2, distance_matrix)

            new_population.extend([child1, child2])

        # Apply elitism (keep the best individual)
        best_individual = min(population, key=lambda x: sum(distance_matrix[x[i]][x[i+1]] for i in range(len(x)-1)))
        new_population[0] = best_individual  # Replace first individual with the best from last gen

        population = new_population
        best_individual_cost = sum(distance_matrix[best_individual[i]][best_individual[i+1]] for i in range(len(best_individual)-1))

        # Store iteration results
        iteration_data.append((iteration, min(fitness_values), max(fitness_values), sum(fitness_values) / len(fitness_values)))

        if best_individual_cost < best_cost:
            best_solution = best_individual
            best_cost = best_individual_cost

    return initial_solution, best_solution, best_cost, iteration_data


def main():
    """Main execution function."""
    if len(sys.argv) != 4:
        print("ERROR: Not enough or too many input arguments.")
        return

    file_path = sys.argv[1]
    num_iterations = int(sys.argv[2])
    mutation_prob = float(sys.argv[3])

    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.")
        return

    start_time = time.time()
    states, distance_matrix = load_data(file_path)
    initial_solution, best_solution, best_cost, iteration_data = genetic_algorithm(states, distance_matrix, num_iterations, mutation_prob)
    execution_time = time.time() - start_time

    # Display results

    print("\nMukesh Siva, A20580319 solution:")
    print(f"Initial state: {states[0]}")
    print(f"\nGenetic Algorithm:")
    print(f"Command Line Parameters: {file_path}, {num_iterations}, {mutation_prob}")
    print(f"Initial solution: {', '.join(initial_solution)}")
    print(f"Final solution: {', '.join(best_solution)}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Complete path cost: {best_cost:.2f}")

if __name__ == "__main__":
    main()

