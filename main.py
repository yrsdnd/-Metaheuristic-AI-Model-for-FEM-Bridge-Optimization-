#!/usr/bin/env python3
"""
FEM Bridge Optimizer
A finite element analysis tool with AI-based structural optimization.
"""

import argparse
import sys

def run_gui():
    import tkinter as tk
    from src.gui import BridgeOptimizerGUI
    
    root = tk.Tk()
    app = BridgeOptimizerGUI(root)
    root.mainloop()

def run_optimization(method='simulated_annealing', iterations=5000):
    from src.optimizer import BridgeOptimizer
    import matplotlib.pyplot as plt
    
    print(f"Running {method} optimization...")
    print(f"Iterations: {iterations}")
    print("-" * 50)
    
    def callback(design_num, iteration, design, objective, temp):
        print(f"Design {design_num} found at iteration {iteration}")
        print(f"  Objective: {objective:.4f}")
        print(f"  Nodes: {len(design['nodes'])}")
        print(f"  Elements: {len(design['elements'])}")
        if temp > 0:
            print(f"  Temperature: {temp:.2f}")
        print()
    
    optimizer = BridgeOptimizer(width=10, height=5)
    
    if method == 'simulated_annealing':
        best_design, best_obj = optimizer.simulated_annealing(
            max_iterations=iterations,
            callback=callback
        )
    else:
        best_design, best_obj = optimizer.genetic_algorithm(
            generations=iterations // 50,
            callback=callback
        )
    
    print("=" * 50)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Objective: {best_obj:.4f}")
    print(f"Final Design: {len(best_design['nodes'])} nodes, {len(best_design['elements'])} elements")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nodes = best_design['nodes']
    elements = best_design['elements']
    
    for elem in elements:
        n1, n2 = elem
        x = [nodes[n1][0], nodes[n2][0]]
        y = [nodes[n1][1], nodes[n2][1]]
        ax.plot(x, y, 'b-', linewidth=2)
    
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ko', markersize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Optimized Bridge Design (Objective: {best_obj:.4f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('optimized_bridge.png', dpi=150, bbox_inches='tight')
    print("Design saved to optimized_bridge.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='FEM Bridge Optimizer - Structural optimization using AI algorithms'
    )
    
    parser.add_argument('--mode', choices=['gui', 'optimize'], default='gui',
                       help='Run mode: gui (interactive) or optimize (command line)')
    parser.add_argument('--method', choices=['simulated_annealing', 'genetic'], 
                       default='simulated_annealing',
                       help='Optimization method')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of iterations for optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        run_gui()
    else:
        run_optimization(args.method, args.iterations)

if __name__ == "__main__":
    main()
