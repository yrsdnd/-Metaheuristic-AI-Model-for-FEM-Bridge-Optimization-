import numpy as np
import random
import copy
from .fem_solver import TrussFEM

class BridgeOptimizer:
    def __init__(self, width=10, height=5, num_bottom_nodes=11, E=210e9, A=0.0001):
        self.width = width
        self.height = height
        self.num_bottom_nodes = num_bottom_nodes
        self.E = E
        self.A = A
        self.best_design = None
        self.best_objective = float('inf')
        self.history = []
        
    def generate_random_bridge(self, num_top_nodes=None):
        if num_top_nodes is None:
            num_top_nodes = random.randint(3, 8)
        
        bottom_nodes = [(i * self.width / (self.num_bottom_nodes - 1), 0) 
                        for i in range(self.num_bottom_nodes)]
        
        top_nodes = []
        for i in range(num_top_nodes):
            x = random.uniform(0.5, self.width - 0.5)
            y = random.uniform(self.height * 0.3, self.height)
            top_nodes.append((x, y))
        
        top_nodes.sort(key=lambda p: p[0])
        
        nodes = bottom_nodes + top_nodes
        elements = []
        
        for i in range(self.num_bottom_nodes - 1):
            elements.append([i, i + 1])
        
        for i in range(len(top_nodes) - 1):
            idx = self.num_bottom_nodes + i
            elements.append([idx, idx + 1])
        
        for i, top_node in enumerate(top_nodes):
            top_idx = self.num_bottom_nodes + i
            tx, ty = top_node
            
            distances = [(j, abs(bottom_nodes[j][0] - tx)) 
                        for j in range(self.num_bottom_nodes)]
            distances.sort(key=lambda x: x[1])
            
            for j, _ in distances[:3]:
                elem = sorted([j, top_idx])
                if elem not in elements:
                    elements.append(elem)
        
        for i in range(len(top_nodes)):
            top_idx = self.num_bottom_nodes + i
            for j in range(i + 2, len(top_nodes)):
                if random.random() < 0.3:
                    other_idx = self.num_bottom_nodes + j
                    elem = sorted([top_idx, other_idx])
                    if elem not in elements:
                        elements.append(elem)
        
        return {'nodes': nodes, 'elements': elements}
    
    def mutate_design(self, design, mutation_strength=0.5):
        new_design = copy.deepcopy(design)
        nodes = list(new_design['nodes'])
        elements = list(new_design['elements'])
        
        mutation_type = random.choice(['move_node', 'add_node', 'remove_node', 'add_element', 'remove_element'])
        
        if mutation_type == 'move_node' and len(nodes) > self.num_bottom_nodes:
            idx = random.randint(self.num_bottom_nodes, len(nodes) - 1)
            x, y = nodes[idx]
            dx = random.gauss(0, mutation_strength)
            dy = random.gauss(0, mutation_strength * 0.5)
            new_x = max(0.1, min(self.width - 0.1, x + dx))
            new_y = max(0.5, min(self.height, y + dy))
            nodes[idx] = (new_x, new_y)
            
        elif mutation_type == 'add_node' and len(nodes) < self.num_bottom_nodes + 10:
            x = random.uniform(0.5, self.width - 0.5)
            y = random.uniform(self.height * 0.3, self.height)
            new_idx = len(nodes)
            nodes.append((x, y))
            
            for i in range(self.num_bottom_nodes):
                if random.random() < 0.3:
                    elements.append(sorted([i, new_idx]))
            
            for i in range(self.num_bottom_nodes, new_idx):
                if random.random() < 0.3:
                    elements.append(sorted([i, new_idx]))
                    
        elif mutation_type == 'remove_node' and len(nodes) > self.num_bottom_nodes + 2:
            idx = random.randint(self.num_bottom_nodes, len(nodes) - 1)
            nodes.pop(idx)
            elements = [[n1 if n1 < idx else n1 - 1, n2 if n2 < idx else n2 - 1] 
                       for n1, n2 in elements if n1 != idx and n2 != idx]
            elements = [sorted(e) for e in elements]
            elements = list(set(tuple(e) for e in elements))
            elements = [list(e) for e in elements]
            
        elif mutation_type == 'add_element':
            if len(nodes) > 1:
                n1 = random.randint(0, len(nodes) - 1)
                n2 = random.randint(0, len(nodes) - 1)
                if n1 != n2:
                    elem = sorted([n1, n2])
                    if elem not in elements:
                        elements.append(elem)
                        
        elif mutation_type == 'remove_element' and len(elements) > self.num_bottom_nodes:
            idx = random.randint(self.num_bottom_nodes - 1, len(elements) - 1)
            elements.pop(idx)
        
        new_design['nodes'] = nodes
        new_design['elements'] = elements
        return new_design
    
    def evaluate_design(self, design, load_value=-20000):
        nodes = design['nodes']
        elements = design['elements']
        
        if len(nodes) < 3 or len(elements) < 2:
            return float('inf')
        
        try:
            fem = TrussFEM(nodes, elements, self.E, self.A)
            
            fixed_dofs = [0, 1, 2 * (self.num_bottom_nodes - 1), 2 * (self.num_bottom_nodes - 1) + 1]
            
            mid_node = self.num_bottom_nodes // 2
            forces = {2 * mid_node + 1: load_value}
            
            displacements = fem.solve(fixed_dofs, forces)
            
            if displacements is None:
                return float('inf')
            
            max_disp = fem.get_max_displacement()
            max_stress = fem.get_max_stress()
            weight = fem.compute_weight()
            
            yield_stress = 250e6
            max_allowed_disp = 0.01
            
            stress_penalty = max(0, (max_stress - yield_stress) / yield_stress) * 100
            disp_penalty = max(0, (max_disp - max_allowed_disp) / max_allowed_disp) * 50
            
            objective = weight / 1000 + stress_penalty + disp_penalty
            
            return objective
            
        except Exception:
            return float('inf')
    
    def simulated_annealing(self, max_iterations=5000, initial_temp=100, cooling_rate=0.995, callback=None):
        current_design = self.generate_random_bridge()
        current_objective = self.evaluate_design(current_design)
        
        self.best_design = copy.deepcopy(current_design)
        self.best_objective = current_objective
        
        temperature = initial_temp
        design_counter = 0
        
        for iteration in range(max_iterations):
            new_design = self.mutate_design(current_design)
            new_objective = self.evaluate_design(new_design)
            
            delta = new_objective - current_objective
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_design = new_design
                current_objective = new_objective
                
                if current_objective < self.best_objective:
                    self.best_design = copy.deepcopy(current_design)
                    self.best_objective = current_objective
                    design_counter += 1
                    
                    if callback:
                        callback(design_counter, iteration, self.best_design, self.best_objective, temperature)
            
            temperature *= cooling_rate
            
            if iteration % 500 == 0:
                self.history.append({
                    'iteration': iteration,
                    'objective': self.best_objective,
                    'temperature': temperature
                })
        
        return self.best_design, self.best_objective
    
    def genetic_algorithm(self, population_size=50, generations=100, mutation_rate=0.3, callback=None):
        population = [self.generate_random_bridge() for _ in range(population_size)]
        fitnesses = [self.evaluate_design(d) for d in population]
        
        design_counter = 0
        
        for generation in range(generations):
            sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
            fitnesses = [f for f, _ in sorted_pop]
            population = [d for _, d in sorted_pop]
            
            if fitnesses[0] < self.best_objective:
                self.best_objective = fitnesses[0]
                self.best_design = copy.deepcopy(population[0])
                design_counter += 1
                
                if callback:
                    callback(design_counter, generation * population_size, self.best_design, self.best_objective, 0)
            
            elite_size = population_size // 5
            new_population = population[:elite_size]
            
            while len(new_population) < population_size:
                parent1 = population[random.randint(0, elite_size - 1)]
                parent2 = population[random.randint(0, elite_size - 1)]
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self.mutate_design(child)
                
                new_population.append(child)
            
            population = new_population
            fitnesses = [self.evaluate_design(d) for d in population]
            
            self.history.append({
                'generation': generation,
                'best_fitness': min(fitnesses),
                'avg_fitness': np.mean(fitnesses)
            })
        
        return self.best_design, self.best_objective
    
    def crossover(self, parent1, parent2):
        child_nodes = list(parent1['nodes'][:self.num_bottom_nodes])
        
        top_nodes1 = parent1['nodes'][self.num_bottom_nodes:]
        top_nodes2 = parent2['nodes'][self.num_bottom_nodes:]
        
        split = len(top_nodes1) // 2
        child_top = list(top_nodes1[:split]) + list(top_nodes2[split:])
        child_nodes.extend(child_top)
        
        child_elements = []
        for i in range(self.num_bottom_nodes - 1):
            child_elements.append([i, i + 1])
        
        for i in range(len(child_top) - 1):
            idx = self.num_bottom_nodes + i
            child_elements.append([idx, idx + 1])
        
        for i, top_node in enumerate(child_top):
            top_idx = self.num_bottom_nodes + i
            tx, ty = top_node
            
            for j in range(self.num_bottom_nodes):
                bx, by = child_nodes[j]
                if abs(bx - tx) < self.width / 3:
                    elem = sorted([j, top_idx])
                    if elem not in child_elements:
                        child_elements.append(elem)
        
        return {'nodes': child_nodes, 'elements': child_elements}


def optimize_bridge(method='simulated_annealing', **kwargs):
    optimizer = BridgeOptimizer(**kwargs)
    
    if method == 'simulated_annealing':
        return optimizer.simulated_annealing(
            max_iterations=kwargs.get('max_iterations', 5000),
            initial_temp=kwargs.get('initial_temp', 100),
            cooling_rate=kwargs.get('cooling_rate', 0.995),
            callback=kwargs.get('callback', None)
        )
    elif method == 'genetic':
        return optimizer.genetic_algorithm(
            population_size=kwargs.get('population_size', 50),
            generations=kwargs.get('generations', 100),
            mutation_rate=kwargs.get('mutation_rate', 0.3),
            callback=kwargs.get('callback', None)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
