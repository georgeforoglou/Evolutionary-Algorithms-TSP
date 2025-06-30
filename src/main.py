import Reporter
import numpy as np
import time
import sys
from numba import njit

import matplotlib.pyplot as plt


# Reinitializes a fraction of the population with random candidates
def restart_population(population, current_data, restart_fraction=0.2):
    num_restarts = int(len(population) * restart_fraction)
    for i in range(num_restarts):
        population[i] = get_random_candidate(current_data)
    return population

# Replaces np.inf in the distance matrix with a large finite value based on the maximum finite distance
def preprocess_graph(distance_matrix, scale_factor=100):
    max_finite_value = np.max(distance_matrix[distance_matrix != np.inf])
    large_value = max_finite_value * scale_factor  # Scaled replacement for np.inf
    processed_matrix = np.where(distance_matrix == np.inf, large_value, distance_matrix)
    return processed_matrix

# Calculates the diversity of the population based on Hamming distance
@njit
def calculate_population_diversity(population):
    num_individuals = len(population)
    diversity_sum = 0
    
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            diversity_sum += np.sum(population[i] != population[j])
    
    # Average diversity
    diversity = diversity_sum / (num_individuals * (num_individuals - 1) / 2)
    return diversity

# Fitness sharing implementation for diversity promotion
@njit
def fitness_sharing(population, current_data, sigma_share):
    num_individuals = len(population)
    shared_fitness = np.zeros(num_individuals)

    # Calculate fitness for each individual
    obj_vals = np.array([obj_fun(individual, current_data)[0] for individual in population])

    for i in range(num_individuals):
        sharing_sum = 0
        for j in range(num_individuals):
            if i != j:
                # Calculate Hamming distance between individuals
                hamming_distance = np.sum(population[i] != population[j])
                # Apply the sharing function
                if hamming_distance < sigma_share:
                    sharing_sum += 1 - (hamming_distance / sigma_share)

        # Adjust fitness based on sharing
        shared_fitness[i] = obj_vals[i] / (1 + sharing_sum)

    return shared_fitness

# 2-opt implementation for LSO
@njit
def two_opt(candidate, distance_matrix, max_iterations=10):
    num_cities = len(candidate) - 1                         # Exclude the return to start city
    best_candidate = candidate.copy()
    best_cost, _ = obj_fun(best_candidate, distance_matrix)

    for _ in range(max_iterations):
        improved = False
        for i in range(1, num_cities - 1):
            for j in range(i + 1, min(i + 10, num_cities)): # Limit to the nearest 10 edges
                # Reverse the segment between i and j
                new_candidate = np.empty_like(best_candidate)
                new_candidate[:i] = best_candidate[:i]
                new_candidate[i:j + 1] = best_candidate[i:j + 1][::-1]
                new_candidate[j + 1:] = best_candidate[j + 1:]
                new_candidate[-1] = new_candidate[0]        # Ensure cycle

                new_cost, _ = obj_fun(new_candidate, distance_matrix)
                if new_cost < best_cost:
                    best_candidate = new_candidate
                    best_cost = new_cost
                    improved = True
                    break  # Early exit to speed up
            if improved:
                break
        if not improved:
            break

    return best_candidate

# Function that returns a random candidate (random permutation). It is not necessarily feasible
@njit
def get_random_candidate(current_data):
	num_cities = len(current_data)
	random_candidate = np.random.permutation(num_cities)
	random_candidate = np.append(random_candidate, random_candidate[0])	
	return random_candidate

# Generate a candidate solution using the Nearest Neighbor heuristic
@njit
def nearest_neighbor_candidate(current_data):
    num_cities = len(current_data)
    start_city = np.random.randint(0, num_cities)  # Start from a random city
    candidate = [start_city]
    unvisited = np.ones(num_cities, dtype=np.bool_)  # Boolean array for unvisited cities
    unvisited[start_city] = False

    while np.any(unvisited):	
        last_city = candidate[-1]

		# Find the nearest unvisited city
        nearest_city = -1
        nearest_distance = np.inf
        for city in range(num_cities):
            if unvisited[city] and current_data[last_city, city] < nearest_distance:
                nearest_city = city
                nearest_distance = current_data[last_city, city]

        candidate.append(nearest_city)

        unvisited[nearest_city] = False

    # Complete the cycle by returning to the start city
    candidate.append(start_city)
	
    return np.array(candidate)

# Function for calculation of a candidate's objective value 
# Returns both the objective function and the total number of np.inf elements for the candidate
@njit
def obj_fun(current_candidate, current_data):
	total_cost = 0
	total_infs = 0

	for current_index in range(len(current_data)):
		current_start_city = current_candidate[current_index]
		current_destination_city = current_candidate[current_index+1]
		current_cost = current_data[current_start_city, current_destination_city]
		
		# Check if the path is infinite
		if(current_cost == np.inf):
			total_infs += 1

		# Add up the total cost of a trip
		total_cost = total_cost + current_cost

	return total_cost, total_infs

@njit
def pmx_crossover(parent1, parent2):
    # Step 1: Ensure input is in correct format (numpy arrays, last element equals the first)
    assert parent1[0] == parent1[-1], "Parent1 must be a closed loop (first == last element)."
    assert parent2[0] == parent2[-1], "Parent2 must be a closed loop (first == last element)."
    
    # Remove the last element (since it's the same as the first) for crossover
    p1 = parent1[:-1]
    p2 = parent2[:-1]
    size = len(p1)
    
    # Step 2: Choose two random crossover points using numpy's random.choice
    cx_point1, cx_point2 = sorted(np.random.choice(size, 2, replace=False))
    
    # Step 3: Initialize offspring with -1 values for clarity
    offspring1 = np.full(size, -1)
    offspring2 = np.full(size, -1)
    
    # Step 4: Copy the segment between crossover points
    offspring1[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
    offspring2[cx_point1:cx_point2] = p2[cx_point1:cx_point2]
    
    # Step 5: Map the values from the segment in the opposite parent
    def pmx_fill(offspring, segment, other_parent):
        for i in range(cx_point1, cx_point2):
            if other_parent[i] not in segment:
                pos = i
                while offspring[pos] != -1:
                    pos = np.where(other_parent == offspring[pos])[0][0]
                offspring[pos] = other_parent[i]
    
    # Fill in the mappings
    pmx_fill(offspring1, p1[cx_point1:cx_point2], p2)
    pmx_fill(offspring2, p2[cx_point1:cx_point2], p1)
    
    # Step 6: Fill in remaining positions with genes from the opposite parent
    offspring1[offspring1 == -1] = p2[offspring1 == -1]
    offspring2[offspring2 == -1] = p1[offspring2 == -1]
    
    # Step 7: Add the starting node to close the loop
    offspring1 = np.append(offspring1, offspring1[0])
    offspring2 = np.append(offspring2, offspring2[0])
    
    return offspring1, offspring2

# Implements the inversion mutation for a candidate solution
@njit
def inversion(current_candidate):
	mutated_cand = np.zeros_like(current_candidate)
	mutated_cand[0:-1] = current_candidate[0:-1]

	# Getting two random index points in the candidate
	res = np.random.choice(len(current_candidate)-1, size=2, replace=False)
	left, right = list(np.sort(res))

	# Inversion in the 2 points
	mutated_cand[left:right+1] = np.flip(current_candidate[left:right+1])
	mutated_cand[-1] = mutated_cand[0]
	return mutated_cand

# k-tournament algorithm
def k_tournament_selection(k, current_population, current_data, return_pop_size, replacement=False):

	# Parameters and initialising arrays.	
	num_cities = len(current_data)
	seed_pop_size = len(current_population)
	selected = np.zeros(shape=(return_pop_size, num_cities+1), dtype=np.int64)
	# In each iteration one candidates is ppicked
	eligibility_flags = np.ones(shape=(seed_pop_size,))		# These will help for the implementation of the replacement=False
	accepted_indices = np.where(eligibility_flags==1)[0]
	for current_iteration in range(return_pop_size):
		if(replacement == False):
			accepted_indices = np.where(eligibility_flags==1)[0]
		sampled_indices = np.random.choice(accepted_indices, size=k, replace=False)			#for Selection replacement=True, for elimination replacement=False
		
		# For the sampled candidates I find the best one:
		best_objective = np.inf
		best_candidate_index = -1
		for current_index in sampled_indices:
			current_obj,_ = obj_fun(current_population[current_index], current_data)
			if(current_obj < best_objective):
				best_objective = current_obj
				best_candidate_index = current_index
		selected[current_iteration] = current_population[best_candidate_index]
		# I change the accepted indices
		if(replacement == False):
			eligibility_flags[best_candidate_index] = 0

	return selected

# This function takes some population and uses descriptive statistics for the objective values (e.g. returning mean-max values, histograms etc)
def population_obj_statistics(current_data, current_population, current_generation):
	# Getting all of the objective values:
	obj_vals = []
	for i in current_population:
		current_obj,_ = obj_fun(i, current_data)
		obj_vals.append(current_obj)

	print(f"Generation: {current_generation}, Mean obj value: {np.mean(obj_vals)}, Best obj value: {np.min(obj_vals)}")

	best_cand = np.argmin(obj_vals)

	return np.mean(obj_vals), np.min(obj_vals), current_population[best_cand]


# Modify the class name to match your student number.
class EA:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		# Parameter initialization
		self.pop_size = 600		                # Population size
		self.num_generations = 1000		        # Number of generations for EA to run
		self.num_parents = 2 * self.pop_size	# Number of parents for recoombination
		self.mutation_probability = 0.1	        # Probability for mutation operation
		self.k = 3		                        # k for k-tournament

		# For convergence plotting
		self.mean_objectives = []
		self.best_objectives = []
		self.time_stamps = []

	# Population initialization
	def initialise_population(self, current_data, pop_size):
		population = []
		nearest_neighbor_fraction = 0.8                     # 80% nearest neighbor, 20% random
		num_nn = int(pop_size * nearest_neighbor_fraction)  # Number of NN-initialised candidates
		num_random = pop_size - num_nn                      # Num of random generatred candidates

		# Nearest Neighbor Initialization
		for _ in range(num_nn):
			current_candidate = nearest_neighbor_candidate(current_data)
			# Refine with 2-opt
			current_candidate = two_opt(current_candidate, current_data)
            # Calculate objective value for candidate
			current_obj, _ = obj_fun(current_candidate, current_data) 
			# Feasibility check
			if current_obj != np.inf:
				population.append(current_candidate)

        # Random Initialization
		while len(population) < pop_size:
			current_candidate = get_random_candidate(current_data)
			# Calculate objective value for candidate
			current_obj, _ = obj_fun(current_candidate, current_data)
			# Feasibility check
			if current_obj != np.inf:
				population.append(current_candidate)

		return np.array(population)
	
    # Selection operation with fitness sharing for diversity promotion
	def selection_with_fitness_sharing(self, population, current_data, return_pop_size, sigma_share, replacement=True):
		
		shared_fitness = fitness_sharing(population, current_data, sigma_share)

		# Rank individuals based on shared fitness
		ranking = np.argsort(np.argsort(shared_fitness)) + 1
		b = len(population)
		a = (1 - 0.99) / ((1 - b) ** 2)
		I_scores = a * (ranking - b) ** 2 + 0.99
		pdf = I_scores / np.sum(I_scores)

		indices = np.random.choice(len(population), size=return_pop_size, replace=replacement, p=pdf)
		return population[indices]
	
	# Crossover operation
	def crossover(self, parents, current_data, technique="pmx"):

		num_parents = len(parents)
		assert num_parents % 2 == 0

		if(technique == "pmx"):
			num_children = len(parents)

		children = np.zeros(shape=(num_children, len(parents[0])), dtype=np.int64)
		for current_index in range(0, num_parents, 2):
			current_parent1 = parents[current_index]
			current_parent2 = parents[current_index+1]
			num_feas_offsprings = 0

			# Constantly creating offsprings until they are feasible
			while(1):
				offspring1, offspring2 = pmx_crossover(current_parent1, current_parent2)
				obj1,_ = obj_fun(offspring1, current_data)
				obj2,_ = obj_fun(offspring2, current_data)
				
				if(obj1 != np.inf):
					children[current_index+num_feas_offsprings] = offspring1
					num_feas_offsprings += 1
					if(num_feas_offsprings >= 2):
						break
				if(obj2 != np.inf):
					children[current_index+num_feas_offsprings] = offspring2
					num_feas_offsprings += 1
					if(num_feas_offsprings >= 2):
						break
		return children

	# Mutation operation with 2-opt refinement
	def mutation(self, augmented_pop, current_data, mutation_probability):
		num_candidates = len(augmented_pop)

		for current_index in range(num_candidates):
			random_exp = np.random.random()
			if(random_exp < mutation_probability):          # This means that I will apply mutation in this candidate
				current_candidate = augmented_pop[current_index]
				while(1):
					current_mutation = inversion(current_candidate)
					mut_obj,_ = obj_fun(current_mutation, current_data)
					if(mut_obj != np.inf):
						augmented_pop[current_index] = current_mutation
						break
					
				# Apply 2-Opt refinement
				augmented_pop[current_index] = two_opt(augmented_pop[current_index], current_data)
				
		return augmented_pop
	
	# def mutation_with_adaptive_rate(self, augmented_pop, current_data, generation, max_generations, diversity, base_rate=0.05, max_rate=0.2):
	# 	num_candidates = len(augmented_pop)
	# 	mutation_probability = adaptive_mutation_rate(generation, max_generations, diversity, base_rate, max_rate)

	# 	for current_index in range(num_candidates):
	# 		if np.random.random() < mutation_probability:
	# 			current_candidate = augmented_pop[current_index]
	# 			mutated_candidate = inversion(current_candidate)
	# 			augmented_pop[current_index] = two_opt(mutated_candidate, current_data)

	# 	return augmented_pop
	
	# Elimination operation
	def elimination(self, augmented_pop, k, current_data, return_pop_size, technique="k_tournament"):

		#Getting the best solution (elitism)
		obj_vals = []
		for current_candidate in augmented_pop:
			current_obj,_ = obj_fun(current_candidate, current_data)
			obj_vals.append(current_obj)
		obj_vals = np.array(obj_vals)
		best_value = np.min(obj_vals)
		best_index = np.argmin(obj_vals)
		best_cand = augmented_pop[best_index]
		#Got the best candidate

		if(technique == "k_tournament"):
			survivors = k_tournament_selection(k, augmented_pop, current_data, return_pop_size)
		survivors[0] = best_cand
		
		return survivors


	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# # Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		# graph_files_path = "../data/"
		# distanceMatrix = read_csv_file(filename, graph_files_path)

		t1=time.time()

		distanceMatrix = preprocess_graph(distanceMatrix)

		# Initialization.
		current_population = self.initialise_population(distanceMatrix, self.pop_size)
		t2=time.time()
		print("Initialization time ", t2-t1)

		duration_array = []
		# sigma_share = 0.1

		# yourConvergenceTestsHere = True
		# while( yourConvergenceTestsHere ):

		# Loop for fixed number of generations
		for generation in range(self.num_generations):
			# meanObjective = 0.0
			# bestObjective = 0.0
			# bestSolution = np.array([1,2,3,4,5])

			# Calculate population diversity
			diversity = calculate_population_diversity(current_population)

			# # Adjust sigma_share based on diversity
			# sigma_share = adjust_sigma_share(10, diversity)

			# Selection
			tsel1=time.time()
			current_parents = self.selection_with_fitness_sharing(current_population, distanceMatrix, self.num_parents, sigma_share=15)
			tsel2=time.time()

			# Recombination
			trecomb1=time.time()
			current_offspring = self.crossover(current_parents, distanceMatrix, "pmx")
			current_augmented_pop = np.vstack([current_population, current_offspring])
			trecomb2=time.time()
			
			# Mutation
			tmut1=time.time()
			current_augmented_pop = self.mutation(current_augmented_pop, distanceMatrix, self.mutation_probability)
			# current_augmented_pop = mutation_with_adaptive_rate(current_augmented_pop, distanceMatrix, generation, self.num_generations, diversity, base_rate=0.05, max_rate=0.2)
			tmut2=time.time()

            # Every 200 generation to promote exploration
			if generation % 200 == 0:
				current_population = restart_population(current_population, distanceMatrix)

			# Elimination
			telim1=time.time()
			current_population = self.elimination(current_augmented_pop, self.k, distanceMatrix, self.pop_size)
			telim2=time.time()

			meanObjective, bestObjective, bestSolution = population_obj_statistics(distanceMatrix, current_population, generation)

			# Store objectives and time for convergence graph
			self.mean_objectives.append(meanObjective)
			self.best_objectives.append(bestObjective)
			self.time_stamps.append(time.time() - t1)

			print("  Diversity of current population", diversity)
			selection_time = tsel2 - tsel1
			print(f"  Selection time: {selection_time:.4f} seconds")
			recombination_time = trecomb2 - trecomb1
			print(f"  Recombination time: {recombination_time:.4f} seconds")
			mutation_time = tmut2 - tmut1
			print(f"  Mutation time: {mutation_time:.4f} seconds")
			elimination_time = telim2 - telim1
			print(f"  Elimination time: {elimination_time:.4f} seconds")
			elapsed_time = time.time() - t1
			remaining_time = 300 - elapsed_time
			print(f"  Remaining time: {remaining_time:.4f} seconds")

			# Collect timing for this generation
			duration_array.append([
				tsel2 - tsel1,       # Selection time
				trecomb2 - trecomb1, # Recombination time
				tmut2 - tmut1,       # Mutation time
				telim2 - telim1      # Elimination time
			])

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0

			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
		
		print("\n")
		print("Best tour length: ", bestObjective)
		print("Best tour: ", bestSolution)
		t3=time.time()

		duration_array = np.array(duration_array)
		average_times = np.mean(duration_array, axis=0)
		
		print("\n")
		print("Initialization time: ", t2-t1)
		print("Average Selection time: ", average_times[0])
		print("Average Recombination time: ", average_times[1])
		print("Average Mutation time: ", average_times[2])
		print("Average Elimination time: ", average_times[3])
		print("\n")
		print("Total time: ", t3-t1)

		# Plot convergence graph
		# self.plot_convergence()

		# Your code here.
		return bestObjective, meanObjective
	
	def plot_convergence(self):
		plt.figure(figsize=(10, 6))
		plt.plot(self.time_stamps, self.mean_objectives, label='Mean Objective', color='blue', alpha=0.7)
		plt.plot(self.time_stamps, self.best_objectives, label='Best Objective', color='red')
		plt.xlabel('Time (seconds)')
		plt.ylabel('Objective Value')
		plt.title('Convergence Plot')
		plt.legend()
		plt.grid(True)
		plt.show()

# @njit
# def adaptive_mutation_rate(generation, max_generations, diversity, base_rate=0.05, max_rate=0.2):
#     """
#     Adjusts mutation rate dynamically based on diversity and improvement rate.
#     """
#     if diversity < 0.1:  # Low diversity or stagnating objective values
#         # Increase mutation for exploration
#         return max_rate
#     else:
#         # Decrease mutation for fine-tuning
#         return base_rate + (max_rate - base_rate) * (1 - generation / max_generations)

# # def adaptive_mutation_rate(generation, max_generations, base_rate=0.05, max_rate=0.3):
# #     """
# #     Adjust mutation rate dynamically based on generation progress.
# #     """
# #     return base_rate + (max_rate - base_rate) * (generation / max_generations)

# @njit
# def adjust_sigma_share(current_sigma, diversity, diversity_threshold=0.1, adjustment_factor=0.05):
#     """
#     Adjusts sigma_share based on population diversity.
#     """
#     if diversity < diversity_threshold:
#         # Low diversity, increase sigma_share
#         current_sigma += adjustment_factor
#     else:
#         # High diversity, decrease sigma_share
#         current_sigma -= adjustment_factor

#     # Ensure sigma_share remains within valid bounds
#     current_sigma = max(0.01, min(current_sigma, 1.0))
#     return current_sigma

if __name__ == '__main__':
	a = EA()

	# Call the file (main.py) in the terminal with an argunment like "tour50.csv"
	# Example: python3 main.py "tour50.csv"
	a.optimize(sys.argv[1])