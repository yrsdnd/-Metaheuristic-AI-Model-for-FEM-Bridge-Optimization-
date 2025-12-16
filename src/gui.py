import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading

from .fem_solver import TrussFEM, run_bridge_analysis
from .optimizer import BridgeOptimizer

class BridgeOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("F.E.M.Boys Project 1 - Bridge Optimizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        self.nodes = []
        self.elements = []
        self.fixed_nodes = []
        self.loads = {}
        self.results = None
        self.optimizer = None
        self.optimization_running = False
        
        self.E = tk.DoubleVar(value=210e9)
        self.A = tk.DoubleVar(value=0.0001)
        
        self.setup_styles()
        self.create_notebook()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TNotebook', background='#1a1a2e')
        style.configure('TNotebook.Tab', 
                       background='#16213e',
                       foreground='white',
                       padding=[20, 10],
                       font=('Helvetica', 11, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', '#e94560')],
                 foreground=[('selected', 'white')])
        
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='white', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))
        style.configure('TLabelframe', background='#1a1a2e', foreground='white')
        style.configure('TLabelframe.Label', background='#1a1a2e', foreground='#e94560', font=('Helvetica', 11, 'bold'))
        
    def create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.create_preprocess_tab()
        self.create_process_tab()
        self.create_postprocess_tab()
        
    def create_preprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Pre Process')
        
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        nodes_frame = ttk.LabelFrame(left_frame, text='Nodes')
        nodes_frame.pack(fill='x', pady=5)
        
        ttk.Label(nodes_frame, text='X:').grid(row=0, column=0, padx=5, pady=5)
        self.node_x = ttk.Entry(nodes_frame, width=10)
        self.node_x.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(nodes_frame, text='Y:').grid(row=0, column=2, padx=5, pady=5)
        self.node_y = ttk.Entry(nodes_frame, width=10)
        self.node_y.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(nodes_frame, text='Generate Node', command=self.add_node).grid(row=1, column=0, columnspan=4, pady=5)
        
        elements_frame = ttk.LabelFrame(left_frame, text='Elements')
        elements_frame.pack(fill='x', pady=5)
        
        ttk.Label(elements_frame, text='End Node 1:').grid(row=0, column=0, padx=5, pady=5)
        self.elem_n1 = ttk.Entry(elements_frame, width=8)
        self.elem_n1.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(elements_frame, text='End Node 2:').grid(row=0, column=2, padx=5, pady=5)
        self.elem_n2 = ttk.Entry(elements_frame, width=8)
        self.elem_n2.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(elements_frame, text='Generate Elements', command=self.add_element).grid(row=1, column=0, columnspan=4, pady=5)
        
        load_frame = ttk.LabelFrame(left_frame, text='Load Design')
        load_frame.pack(fill='x', pady=5)
        ttk.Button(load_frame, text='Load from File', command=self.load_design).pack(pady=5)
        ttk.Button(load_frame, text='Generate Sample Bridge', command=self.generate_sample_bridge).pack(pady=5)
        ttk.Button(load_frame, text='Clear All', command=self.clear_all).pack(pady=5)
        
        center_frame = ttk.Frame(tab)
        center_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        self.pre_fig = Figure(figsize=(8, 6), facecolor='#0f0f23')
        self.pre_ax = self.pre_fig.add_subplot(111)
        self.pre_ax.set_facecolor('#0f0f23')
        self.pre_ax.tick_params(colors='white')
        self.pre_ax.spines['bottom'].set_color('orange')
        self.pre_ax.spines['top'].set_color('orange')
        self.pre_ax.spines['left'].set_color('orange')
        self.pre_ax.spines['right'].set_color('orange')
        self.pre_ax.grid(True, color='orange', alpha=0.3)
        
        self.pre_canvas = FigureCanvasTkAgg(self.pre_fig, master=center_frame)
        self.pre_canvas.draw()
        self.pre_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar_frame = ttk.Frame(center_frame)
        toolbar_frame.pack(fill='x')
        NavigationToolbar2Tk(self.pre_canvas, toolbar_frame)
        
    def create_process_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Process')
        
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        bc_frame = ttk.LabelFrame(left_frame, text='Boundary Conditions')
        bc_frame.pack(fill='x', pady=5)
        
        ttk.Label(bc_frame, text='Node Number:').grid(row=0, column=0, padx=5, pady=5)
        self.bc_node = ttk.Entry(bc_frame, width=8)
        self.bc_node.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(bc_frame, text='Direction:').grid(row=1, column=0, padx=5, pady=5)
        self.bc_dir = tk.StringVar(value='X')
        ttk.Radiobutton(bc_frame, text='X', variable=self.bc_dir, value='X').grid(row=1, column=1)
        ttk.Radiobutton(bc_frame, text='Y', variable=self.bc_dir, value='Y').grid(row=1, column=2)
        
        ttk.Label(bc_frame, text='BC Type:').grid(row=2, column=0, padx=5, pady=5)
        self.bc_type = tk.StringVar(value='Displacement')
        ttk.Radiobutton(bc_frame, text='Force', variable=self.bc_type, value='Force').grid(row=2, column=1)
        ttk.Radiobutton(bc_frame, text='Displacement', variable=self.bc_type, value='Displacement').grid(row=2, column=2)
        
        ttk.Label(bc_frame, text='Magnitude:').grid(row=3, column=0, padx=5, pady=5)
        self.bc_mag = ttk.Entry(bc_frame, width=10)
        self.bc_mag.insert(0, '-20000')
        self.bc_mag.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Button(bc_frame, text='Apply', command=self.apply_bc).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(bc_frame, text='Clear All', command=self.clear_bcs).grid(row=4, column=2, pady=5)
        
        mat_frame = ttk.LabelFrame(left_frame, text='Material Properties')
        mat_frame.pack(fill='x', pady=5)
        
        ttk.Label(mat_frame, text="Young's Modulus (E):").grid(row=0, column=0, padx=5, pady=5)
        self.e_entry = ttk.Entry(mat_frame, width=15)
        self.e_entry.insert(0, '210e09')
        self.e_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(mat_frame, text='Area (A):').grid(row=1, column=0, padx=5, pady=5)
        self.a_entry = ttk.Entry(mat_frame, width=15)
        self.a_entry.insert(0, '0.0001')
        self.a_entry.grid(row=1, column=1, padx=5, pady=5)
        
        run_btn = tk.Button(left_frame, text='Run Analysis', bg='#e94560', fg='white',
                           font=('Helvetica', 12, 'bold'), command=self.run_analysis)
        run_btn.pack(pady=20)
        
        center_frame = ttk.Frame(tab)
        center_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        self.proc_fig = Figure(figsize=(8, 6), facecolor='#0f0f23')
        self.proc_ax = self.proc_fig.add_subplot(111)
        self.setup_dark_axes(self.proc_ax)
        
        self.proc_canvas = FigureCanvasTkAgg(self.proc_fig, master=center_frame)
        self.proc_canvas.draw()
        self.proc_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_postprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Post Process')
        
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        cmap_frame = ttk.LabelFrame(controls_frame, text='Colormap')
        cmap_frame.pack(side='left', padx=10)
        
        self.cmap_var = tk.StringVar(value='jet')
        for cmap in ['Jet', 'Copper', 'Cool']:
            ttk.Radiobutton(cmap_frame, text=cmap, variable=self.cmap_var, 
                           value=cmap.lower(), command=self.update_postprocess).pack(anchor='w')
        
        scale_frame = ttk.LabelFrame(controls_frame, text='Scale Factor')
        scale_frame.pack(side='left', padx=10)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(scale_frame, from_=0.1, to=100, variable=self.scale_var,
                                      orient='horizontal', length=150, command=lambda x: self.update_postprocess())
        self.scale_slider.pack(padx=5, pady=5)
        self.scale_label = ttk.Label(scale_frame, text='1.00')
        self.scale_label.pack()
        
        display_frame = ttk.LabelFrame(controls_frame, text='Display')
        display_frame.pack(side='left', padx=10)
        
        self.show_ux = tk.BooleanVar(value=True)
        self.show_uy = tk.BooleanVar(value=True)
        self.show_undeformed = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(display_frame, text='U_x', variable=self.show_ux, command=self.update_postprocess).pack(anchor='w')
        ttk.Checkbutton(display_frame, text='U_y', variable=self.show_uy, command=self.update_postprocess).pack(anchor='w')
        ttk.Checkbutton(display_frame, text='Show Undeformed Shape', variable=self.show_undeformed, command=self.update_postprocess).pack(anchor='w')
        
        ttk.Button(controls_frame, text='Reset View', command=self.reset_postprocess).pack(side='right', padx=10)
        
        plots_frame = ttk.Frame(tab)
        plots_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_plot = ttk.Frame(plots_frame)
        left_plot.pack(side='left', fill='both', expand=True)
        
        self.post_fig1 = Figure(figsize=(6, 5), facecolor='#0f0f23')
        self.post_ax1 = self.post_fig1.add_subplot(111)
        self.setup_dark_axes(self.post_ax1)
        self.post_ax1.set_title('Displacement: U_y', color='white')
        
        self.post_canvas1 = FigureCanvasTkAgg(self.post_fig1, master=left_plot)
        self.post_canvas1.draw()
        self.post_canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        right_plot = ttk.Frame(plots_frame)
        right_plot.pack(side='right', fill='both', expand=True)
        
        self.post_fig2 = Figure(figsize=(6, 5), facecolor='#0f0f23')
        self.post_ax2 = self.post_fig2.add_subplot(111)
        self.setup_dark_axes(self.post_ax2)
        self.post_ax2.set_title('Stress Distribution', color='white')
        
        self.post_canvas2 = FigureCanvasTkAgg(self.post_fig2, master=right_plot)
        self.post_canvas2.draw()
        self.post_canvas2.get_tk_widget().pack(fill='both', expand=True)
        
    def setup_dark_axes(self, ax):
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('orange')
        ax.grid(True, color='orange', alpha=0.3)
        
    def add_node(self):
        try:
            x = float(self.node_x.get())
            y = float(self.node_y.get())
            self.nodes.append((x, y))
            self.update_preprocess_plot()
            self.node_x.delete(0, tk.END)
            self.node_y.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Enter valid coordinates")
            
    def add_element(self):
        try:
            n1 = int(self.elem_n1.get()) - 1
            n2 = int(self.elem_n2.get()) - 1
            if 0 <= n1 < len(self.nodes) and 0 <= n2 < len(self.nodes):
                self.elements.append([n1, n2])
                self.update_preprocess_plot()
                self.elem_n1.delete(0, tk.END)
                self.elem_n2.delete(0, tk.END)
            else:
                messagebox.showerror("Error", "Invalid node indices")
        except ValueError:
            messagebox.showerror("Error", "Enter valid node numbers")
            
    def generate_sample_bridge(self):
        self.nodes = []
        self.elements = []
        
        for i in range(11):
            self.nodes.append((i, 0))
        
        for i in range(1, 10):
            if i % 2 == 1:
                self.nodes.append((i, 2 + np.random.uniform(-0.5, 0.5)))
        
        for i in range(10):
            self.elements.append([i, i + 1])
        
        top_start = 11
        for i in range(4):
            self.elements.append([top_start + i, top_start + i + 1])
        
        top_idx = 11
        for i in range(1, 10, 2):
            self.elements.append([i, top_idx])
            if i > 1:
                self.elements.append([i - 1, top_idx])
            if i < 9:
                self.elements.append([i + 1, top_idx])
            top_idx += 1
        
        self.fixed_nodes = [0, 10]
        self.loads = {5: -20000}
        
        self.update_preprocess_plot()
        messagebox.showinfo("Info", f"Generated bridge with {len(self.nodes)} nodes and {len(self.elements)} elements")
        
    def load_design(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            try:
                self.nodes = []
                self.elements = []
                with open(filename, 'r') as f:
                    section = None
                    for line in f:
                        line = line.strip()
                        if line.startswith('Nodes:'):
                            section = 'nodes'
                            continue
                        elif line.startswith('Elements:'):
                            section = 'elements'
                            continue
                        elif not line or line.startswith('#'):
                            continue
                        
                        if section == 'nodes':
                            parts = line.split(',')
                            if len(parts) >= 3:
                                x = float(parts[1])
                                y = float(parts[2])
                                self.nodes.append((x, y))
                        elif section == 'elements':
                            parts = line.split(',')
                            if len(parts) >= 3:
                                n1 = int(parts[1]) - 1
                                n2 = int(parts[2]) - 1
                                self.elements.append([n1, n2])
                
                self.update_preprocess_plot()
                messagebox.showinfo("Success", f"Loaded {len(self.nodes)} nodes and {len(self.elements)} elements")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                
    def clear_all(self):
        self.nodes = []
        self.elements = []
        self.fixed_nodes = []
        self.loads = {}
        self.update_preprocess_plot()
        
    def update_preprocess_plot(self):
        self.pre_ax.clear()
        self.setup_dark_axes(self.pre_ax)
        
        for elem in self.elements:
            n1, n2 = elem
            if n1 < len(self.nodes) and n2 < len(self.nodes):
                x = [self.nodes[n1][0], self.nodes[n2][0]]
                y = [self.nodes[n1][1], self.nodes[n2][1]]
                self.pre_ax.plot(x, y, 'c-', linewidth=2)
        
        for i, (x, y) in enumerate(self.nodes):
            self.pre_ax.plot(x, y, 'co', markersize=8)
            self.pre_ax.annotate(str(i + 1), (x, y), textcoords="offset points", 
                               xytext=(5, 5), color='white', fontsize=8)
        
        for node in self.fixed_nodes:
            if node < len(self.nodes):
                x, y = self.nodes[node]
                self.pre_ax.plot(x, y, 'g^', markersize=15)
        
        for node, force in self.loads.items():
            if node < len(self.nodes):
                x, y = self.nodes[node]
                self.pre_ax.annotate('', xy=(x, y + force/abs(force) * 0.5), xytext=(x, y),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        self.pre_ax.set_xlabel('X (m)', color='white')
        self.pre_ax.set_ylabel('Y (m)', color='white')
        self.pre_ax.set_aspect('equal')
        self.pre_canvas.draw()
        
    def apply_bc(self):
        try:
            node = int(self.bc_node.get()) - 1
            magnitude = float(self.bc_mag.get())
            
            if self.bc_type.get() == 'Displacement':
                if node not in self.fixed_nodes:
                    self.fixed_nodes.append(node)
            else:
                self.loads[node] = magnitude
                
            self.update_preprocess_plot()
        except ValueError:
            messagebox.showerror("Error", "Enter valid values")
            
    def clear_bcs(self):
        self.fixed_nodes = []
        self.loads = {}
        self.update_preprocess_plot()
        
    def run_analysis(self):
        if len(self.nodes) < 2 or len(self.elements) < 1:
            messagebox.showerror("Error", "Define nodes and elements first")
            return
            
        try:
            E = float(self.e_entry.get())
            A = float(self.a_entry.get())
            
            fem = TrussFEM(self.nodes, self.elements, E, A)
            
            fixed_dofs = []
            for node in self.fixed_nodes:
                fixed_dofs.extend([2*node, 2*node + 1])
            
            if not fixed_dofs:
                fixed_dofs = [0, 1, 2*(len(self.nodes)-1), 2*(len(self.nodes)-1) + 1]
            
            forces = {}
            for node, force in self.loads.items():
                forces[2*node + 1] = force
            
            if not forces:
                mid_node = len(self.nodes) // 2
                forces = {2*mid_node + 1: -20000}
            
            displacements = fem.solve(fixed_dofs, forces)
            
            if displacements is None:
                messagebox.showerror("Error", "Analysis failed - singular stiffness matrix")
                return
            
            self.results = {
                'fem': fem,
                'displacements': displacements,
                'stresses': fem.stresses
            }
            
            self.update_process_plot()
            self.update_postprocess()
            
            max_disp = fem.get_max_displacement()
            max_stress = fem.get_max_stress()
            weight = fem.compute_weight()
            
            messagebox.showinfo("Analysis Complete", 
                              f"Max Displacement: {max_disp:.6f} m\n"
                              f"Max Stress: {max_stress/1e6:.2f} MPa\n"
                              f"Weight: {weight:.2f} N")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def update_process_plot(self):
        if self.results is None:
            return
            
        self.proc_ax.clear()
        self.setup_dark_axes(self.proc_ax)
        
        fem = self.results['fem']
        disp = self.results['displacements']
        scale = 50
        
        for elem in self.elements:
            n1, n2 = elem
            x = [self.nodes[n1][0], self.nodes[n2][0]]
            y = [self.nodes[n1][1], self.nodes[n2][1]]
            self.proc_ax.plot(x, y, 'orange', linewidth=1, alpha=0.5)
        
        for i, elem in enumerate(self.elements):
            n1, n2 = elem
            x1 = self.nodes[n1][0] + disp[2*n1] * scale
            y1 = self.nodes[n1][1] + disp[2*n1 + 1] * scale
            x2 = self.nodes[n2][0] + disp[2*n2] * scale
            y2 = self.nodes[n2][1] + disp[2*n2 + 1] * scale
            self.proc_ax.plot([x1, x2], [y1, y2], 'c-', linewidth=2)
        
        for i, (x, y) in enumerate(self.nodes):
            dx = disp[2*i] * scale
            dy = disp[2*i + 1] * scale
            self.proc_ax.plot(x + dx, y + dy, 'co', markersize=6)
        
        self.proc_ax.set_xlabel('X (m)', color='white')
        self.proc_ax.set_ylabel('Y (m)', color='white')
        self.proc_ax.set_aspect('equal')
        self.proc_canvas.draw()
        
    def update_postprocess(self):
        if self.results is None:
            return
            
        scale = self.scale_var.get()
        self.scale_label.config(text=f'{scale:.2f}')
        cmap = self.cmap_var.get()
        
        fem = self.results['fem']
        disp = self.results['displacements']
        stresses = self.results['stresses']
        
        self.post_ax1.clear()
        self.setup_dark_axes(self.post_ax1)
        
        if self.show_undeformed.get():
            for elem in self.elements:
                n1, n2 = elem
                x = [self.nodes[n1][0], self.nodes[n2][0]]
                y = [self.nodes[n1][1], self.nodes[n2][1]]
                self.post_ax1.plot(x, y, 'orange', linewidth=1, alpha=0.3)
        
        disp_mag = np.sqrt(disp[0::2]**2 + disp[1::2]**2)
        norm = plt.Normalize(vmin=disp_mag.min(), vmax=disp_mag.max())
        
        for i, elem in enumerate(self.elements):
            n1, n2 = elem
            x1 = self.nodes[n1][0] + disp[2*n1] * scale
            y1 = self.nodes[n1][1] + disp[2*n1 + 1] * scale
            x2 = self.nodes[n2][0] + disp[2*n2] * scale
            y2 = self.nodes[n2][1] + disp[2*n2 + 1] * scale
            
            avg_disp = (disp_mag[n1] + disp_mag[n2]) / 2
            color = plt.cm.get_cmap(cmap)(norm(avg_disp))
            self.post_ax1.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.post_fig1.colorbar(sm, ax=self.post_ax1, label='Displacement (m)')
        
        self.post_ax1.set_title('Displacement: U_y', color='white')
        self.post_ax1.set_aspect('equal')
        self.post_canvas1.draw()
        
        self.post_ax2.clear()
        self.setup_dark_axes(self.post_ax2)
        
        if self.show_undeformed.get():
            for elem in self.elements:
                n1, n2 = elem
                x = [self.nodes[n1][0], self.nodes[n2][0]]
                y = [self.nodes[n1][1], self.nodes[n2][1]]
                self.post_ax2.plot(x, y, 'orange', linewidth=1, alpha=0.3)
        
        stress_norm = plt.Normalize(vmin=np.abs(stresses).min(), vmax=np.abs(stresses).max())
        
        for i, elem in enumerate(self.elements):
            n1, n2 = elem
            x1 = self.nodes[n1][0] + disp[2*n1] * scale
            y1 = self.nodes[n1][1] + disp[2*n1 + 1] * scale
            x2 = self.nodes[n2][0] + disp[2*n2] * scale
            y2 = self.nodes[n2][1] + disp[2*n2 + 1] * scale
            
            color = plt.cm.get_cmap(cmap)(stress_norm(np.abs(stresses[i])))
            self.post_ax2.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        
        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=stress_norm)
        sm2.set_array([])
        self.post_fig2.colorbar(sm2, ax=self.post_ax2, label='Stress (Pa)')
        
        self.post_ax2.set_title('Stress Distribution', color='white')
        self.post_ax2.set_aspect('equal')
        self.post_canvas2.draw()
        
    def reset_postprocess(self):
        self.scale_var.set(1.0)
        self.cmap_var.set('jet')
        self.show_undeformed.set(True)
        self.update_postprocess()


def main():
    root = tk.Tk()
    app = BridgeOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
