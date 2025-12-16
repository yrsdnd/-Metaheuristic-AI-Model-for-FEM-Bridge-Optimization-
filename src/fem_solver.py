import numpy as np

class TrussFEM:
    def __init__(self, nodes, elements, E=210e9, A=0.0001):
        self.nodes = np.array(nodes)
        self.elements = np.array(elements, dtype=int)
        self.E = E
        self.A = A
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        self.K_global = None
        self.displacements = None
        self.forces = None
        self.stresses = None
        
    def element_stiffness(self, node1_idx, node2_idx):
        x1, y1 = self.nodes[node1_idx]
        x2, y2 = self.nodes[node2_idx]
        
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if L == 0:
            return np.zeros((4, 4))
        
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        
        k = (self.E * self.A / L) * np.array([
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c,  c*s],
            [-c*s, -s*s, c*s,  s*s]
        ])
        return k
    
    def assemble_global_stiffness(self):
        ndof = 2 * self.num_nodes
        self.K_global = np.zeros((ndof, ndof))
        
        for elem in self.elements:
            n1, n2 = elem[0], elem[1]
            k_elem = self.element_stiffness(n1, n2)
            
            dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            
            for i in range(4):
                for j in range(4):
                    self.K_global[dof_map[i], dof_map[j]] += k_elem[i, j]
        
        return self.K_global
    
    def apply_boundary_conditions(self, fixed_dofs, forces):
        ndof = 2 * self.num_nodes
        F = np.zeros(ndof)
        
        for dof, force in forces.items():
            F[dof] = force
        
        free_dofs = [i for i in range(ndof) if i not in fixed_dofs]
        
        K_reduced = self.K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F[free_dofs]
        
        return K_reduced, F_reduced, free_dofs
    
    def solve(self, fixed_dofs, forces):
        self.assemble_global_stiffness()
        K_reduced, F_reduced, free_dofs = self.apply_boundary_conditions(fixed_dofs, forces)
        
        try:
            U_reduced = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            return None
        
        ndof = 2 * self.num_nodes
        self.displacements = np.zeros(ndof)
        for i, dof in enumerate(free_dofs):
            self.displacements[dof] = U_reduced[i]
        
        self.forces = self.K_global @ self.displacements
        self.compute_stresses()
        
        return self.displacements
    
    def compute_stresses(self):
        self.stresses = np.zeros(self.num_elements)
        
        for idx, elem in enumerate(self.elements):
            n1, n2 = elem[0], elem[1]
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            
            L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if L == 0:
                continue
                
            c = (x2 - x1) / L
            s = (y2 - y1) / L
            
            u1x = self.displacements[2*n1]
            u1y = self.displacements[2*n1 + 1]
            u2x = self.displacements[2*n2]
            u2y = self.displacements[2*n2 + 1]
            
            delta_L = (u2x - u1x) * c + (u2y - u1y) * s
            strain = delta_L / L
            self.stresses[idx] = self.E * strain
        
        return self.stresses
    
    def get_max_displacement(self):
        if self.displacements is None:
            return 0
        disp_mag = np.sqrt(self.displacements[0::2]**2 + self.displacements[1::2]**2)
        return np.max(disp_mag)
    
    def get_max_stress(self):
        if self.stresses is None:
            return 0
        return np.max(np.abs(self.stresses))
    
    def compute_weight(self, density=7850):
        total_weight = 0
        for elem in self.elements:
            n1, n2 = elem[0], elem[1]
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_weight += density * self.A * L * 9.81
        return total_weight


def run_bridge_analysis(nodes, elements, fixed_nodes, load_node, load_value, E=210e9, A=0.0001):
    fem = TrussFEM(nodes, elements, E, A)
    
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2*node, 2*node + 1])
    
    forces = {2*load_node + 1: load_value}
    
    displacements = fem.solve(fixed_dofs, forces)
    
    if displacements is None:
        return None
    
    results = {
        'displacements': displacements,
        'stresses': fem.stresses,
        'max_displacement': fem.get_max_displacement(),
        'max_stress': fem.get_max_stress(),
        'weight': fem.compute_weight()
    }
    
    return results
